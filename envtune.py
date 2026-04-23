"""
hybrid_tune.py  -  merged auto_tune (Sniffleupagus) + envtune (hasj) + handshake-maximizing extras

What it does
------------
1. PERSONALITY TUNING (from envtune):
   - EMA-smoothed observation of AP density, handshake rate, reward, missed-interaction rate
   - Bounded adjustments to min_rssi, hop_recon_time, recon_time, max_interactions
   - Blind-panic fallback when recon is failing
   - Reward-revert: if a tweak drops reward, roll back automatically
   - Best-settings memory: biases future tweaks toward the best historical personality
   - Persistent state across reboots (lifetime handshakes, per-channel HS counts, best settings)

2. CHANNEL SCHEDULING (from auto_tune, improved):
   - Always keeps active channels (channels where APs were just seen)
   - Adds N extra channels per epoch: 70% priority (weighted by historical HS yield)
     + 30% pure exploration
   - Dead-channel cooldown: channels with >5 consecutive empty scans are skipped
   - Per-channel session + lifetime statistics (APs, assocs, deauths, handshakes, clients)

3. HANDSHAKE MAXIMIZERS:
   - Tracks per-AP attack count (AT_attacks) so you can see which APs are unresponsive
   - Tracks captured APs in a session set so stats can show unique pwns
   - Bettercap client-new event bumps a 'Clients' chisto -> influences priority scoring
     (channels with recent client activity get extra weight)
   - Web UI at /plugins/hybrid_tune/ with status, per-channel stats, and top-pwned APs
"""

import html
import json
import logging
import os
import random
import threading
import time
from collections import defaultdict

import pwnagotchi.plugins as plugins
import pwnagotchi.utils
from flask import render_template_string


class HybridTune(plugins.Plugin):
    __author__ = 'adi1708 (merged from Sniffleupagus auto_tune + adi1708 envtune)'
    __version__ = '1.0.0'
    __license__ = 'MIT'
    __description__ = (
        'Channel scheduler + environment-aware personality tuner with '
        'handshake-maximizing strategies.'
    )

    STATE_PATH = '/etc/pwnagotchi/hybrid_tune_state.json'

    # (min, max) bounds for personality parameters we tune
    BOUNDS = {
        'min_rssi':         (-85, -65),
        'hop_recon_time':   (4, 15),
        'recon_time':       (15, 45),
        'max_interactions': (2, 6),
    }

    DEFAULTS = {
        # --- tuning (from envtune) ---
        'ema_alpha':             0.35,   # EMA smoothing factor
        'warmup_epochs':         3,      # ignore tuning for first N epochs
        'hs_rate_low':           0.15,   # below this = poor capture
        'hs_rate_high':          0.40,   # above this = great capture
        'dense_aps':             25,     # >= this many APs = crowded env
        'sparse_aps':            8,      # <= this many APs = quiet env
        'blind_panic_epochs':    3,      # N blind epochs triggers safe-preset
        'reward_drop_threshold': 0.25,   # reward drop this much -> revert
        'save_every_n_epochs':   5,
        'best_bias_weight':      0.15,   # how hard to pull toward best-known

        # --- channel scheduling (new) ---
        'extra_channels':           3,   # extras added to active channels per epoch
        'priority_channel_weight':  0.70,  # fraction of extras picked by HS history
        'dead_channel_cooldown':    5,   # skip channel after N empty scans

        # --- ap targeting (new) ---
        'show_hidden':   False,
        'reset_history': True,

        # optional: restrict_channels = [1, 6, 11, 36, 40, ...]
    }

    # -------------------- init & state --------------------

    def __init__(self):
        self.cfg = dict(self.DEFAULTS)

        # EMA-smoothed signals
        self.ema = {'aps': None, 'hs_rate': None, 'reward': None, 'missed_rate': None}
        self.epochs_seen = 0
        self.epochs_since_save = 0

        # Persistent stats
        self.session_start = time.time()
        self.lifetime_handshakes = 0
        self.channel_hs = defaultdict(int)  # lifetime HS count per channel
        self.best_reward = None
        self.best_settings = None

        # Adjustment tracking (for revert logic)
        self.last_adjust = None
        self.reward_before_adjust = None

        # Channel scheduling
        self._active_channels = []
        self._unscanned_channels = []
        self._dead_channel_counter = defaultdict(int)  # consecutive empty scans

        # AP tracking
        self._known_aps = {}        # normalized-name-mac -> AP dict with counters
        self._captured_aps = set()  # apIDs we got a handshake from this session

        # Session statistics
        self._chistos = {'_all_actions': {-1: 0}}   # {stat_name: {channel: count, -1: total}}
        self._histogram = {'loops': 0}

        # Plugin wiring
        self.ep_data = {}
        self.last_shake = {'time': time.time()}
        self._agent = None
        self._ui = None
        self._orig_mode = 'AUTO'

        self._lock = threading.Lock()

    # -------------------- helpers --------------------

    @staticmethod
    def normalize(name):
        if not name:
            return 'EMPTY'
        if name == '<hidden>':
            return 'HIDDEN'
        return str.lower(''.join(c for c in str(name) if c.isalnum()))

    def _ap_id(self, ap):
        return self.normalize(ap.get('hostname', '')) + '-' + self.normalize(ap.get('mac', ''))

    def _clamp(self, key, v):
        lo, hi = self.BOUNDS[key]
        return max(lo, min(hi, int(round(v))))

    def _inc_chisto(self, stat, channel, count=1):
        if stat not in self._chistos:
            self._chistos[stat] = {-1: 0}
        self._chistos[stat][channel] = self._chistos[stat].get(channel, 0) + count
        self._chistos[stat][-1] += count
        aa = self._chistos['_all_actions']
        aa[channel] = aa.get(channel, 0) + count
        aa[-1] += count

    # -------------------- state persistence --------------------

    def _load_state(self):
        try:
            if not os.path.exists(self.STATE_PATH):
                return
            with open(self.STATE_PATH) as f:
                st = json.load(f)
            self.ema.update({k: v for k, v in (st.get('ema') or {}).items() if k in self.ema})
            self.lifetime_handshakes = int(st.get('lifetime_handshakes', 0))
            ch = st.get('channel_hs') or {}
            self.channel_hs = defaultdict(int, {int(k): int(v) for k, v in ch.items()})
            self.best_reward = st.get('best_reward')
            self.best_settings = st.get('best_settings')
            logging.info(
                f"[hybrid_tune] state loaded lifetime_hs={self.lifetime_handshakes} "
                f"best_reward={self.best_reward}"
            )
        except Exception as e:
            logging.warning(f"[hybrid_tune] state load failed: {e}")

    def _save_state(self):
        try:
            with self._lock:
                st = {
                    'ema': self.ema,
                    'lifetime_handshakes': self.lifetime_handshakes,
                    'channel_hs': dict(self.channel_hs),
                    'best_reward': self.best_reward,
                    'best_settings': self.best_settings,
                    'saved_at': time.time(),
                }
            tmp = self.STATE_PATH + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(st, f)
            os.replace(tmp, self.STATE_PATH)
        except Exception as e:
            logging.warning(f"[hybrid_tune] state save failed: {e}")

    def _maybe_save(self):
        if self.epochs_since_save >= self.cfg['save_every_n_epochs']:
            self._save_state()
            self.epochs_since_save = 0

    # -------------------- EMA tuning core --------------------

    def _ema_update(self, key, value):
        a = self.cfg['ema_alpha']
        prev = self.ema.get(key)
        if prev is None or (prev == 0 and self.epochs_seen < self.cfg['warmup_epochs']):
            new = float(value)
        else:
            new = a * float(value) + (1 - a) * prev
        self.ema[key] = new
        return new

    def _bias_toward_best(self, proposed):
        if not self.best_settings:
            return proposed
        w = self.cfg['best_bias_weight']
        out = dict(proposed)
        for k in self.BOUNDS:
            if k in self.best_settings:
                blended = (1 - w) * proposed[k] + w * self.best_settings[k]
                out[k] = self._clamp(k, blended)
        return out

    # -------------------- channel priority scoring --------------------

    def _channel_score(self, ch):
        """Score a channel based on historical productivity.

        Factors:
          - lifetime handshakes on channel (heaviest)
          - session-level deauths/associations (activity)
          - recent client presence (bonus)
        """
        hs = self.channel_hs.get(ch, 0)
        actions = self._chistos.get('_all_actions', {}).get(ch, 0)
        clients = self._chistos.get('Clients', {}).get(ch, 0)
        return hs * 3.0 + actions * 0.3 + clients * 1.5 + 0.01  # epsilon so zeros can still be picked

    def _pick_weighted(self, pool, n):
        """Weighted random selection without replacement."""
        if not pool or n <= 0:
            return []
        candidates = [(c, self._channel_score(c)) for c in pool]
        total = sum(s for _, s in candidates)
        picks = []
        while len(picks) < n and candidates and total > 0:
            r = random.random() * total
            acc = 0.0
            for i, (c, s) in enumerate(candidates):
                acc += s
                if acc >= r:
                    picks.append(c)
                    total -= s
                    candidates.pop(i)
                    break
        return picks

    # -------------------- plugin lifecycle --------------------

    def on_loaded(self):
        try:
            user = self.options or {}
            self.cfg.update({k: v for k, v in user.items() if k in self.DEFAULTS})
        except Exception:
            pass
        self._load_state()
        logging.info(
            f"[hybrid_tune] v{self.__version__} loaded "
            f"alpha={self.cfg['ema_alpha']} "
            f"lifetime_hs={self.lifetime_handshakes} "
            f"best_reward={self.best_reward}"
        )

    def on_ready(self, agent):
        self._agent = agent
        if self.cfg.get('reset_history', True):
            try:
                self._agent._history = {}
                self._agent.run("wifi.recon clear")
                self._agent.run("wifi.clear")
                channels = agent._config['personality'].get('channels', [1, 6, 11])
                self._agent.run("wifi.recon.channel %s" % (','.join(map(str, channels))))
            except Exception as e:
                logging.warning(f"[hybrid_tune] history reset failed: {e}")

        if agent._config.get('ai', {}).get('enabled', False):
            logging.info("[hybrid_tune] inactive while AI mode is enabled.")
        else:
            logging.info("[hybrid_tune] active - cfg=%s" % repr(self.cfg))

    # -------------------- AP tracking --------------------

    def _mark_ap_seen(self, ap, context=None):
        try:
            apID = self._ap_id(ap)
            channel = ap.get('channel', 0)
            tag = 'AT_' + context if context else 'AT_seen'

            if apID not in self._known_aps:
                self._known_aps[apID] = dict(ap)
                self._known_aps[apID].update({
                    'AT_seen': 1,
                    'AT_visible': True,
                    'AT_attacks': 0,
                    tag: 1,
                })
                self._inc_chisto('Unique APs', channel)
                self._inc_chisto('Current APs', channel)
            else:
                for k in ap:
                    self._known_aps[apID][k] = ap[k]
                if not self._known_aps[apID].get('AT_visible', True):
                    self._known_aps[apID]['AT_visible'] = True
                    self._known_aps[apID]['AT_seen'] = self._known_aps[apID].get('AT_seen', 0) + 1
                    self._inc_chisto('Current APs', channel)
                self._known_aps[apID][tag] = self._known_aps[apID].get(tag, 0) + 1

            self._known_aps[apID]['AT_lastseen'] = time.time()
            return True
        except Exception as e:
            logging.exception(e)
            return False

    # -------------------- event callbacks --------------------

    def on_handshake(self, agent, filename, access_point, client_station):
        try:
            ch = 0
            if isinstance(access_point, dict):
                ch = int(access_point.get('channel', 0) or 0)
                apID = self._ap_id(access_point)
                self._captured_aps.add(apID)
                self._mark_ap_seen(access_point, 'handshake')
            with self._lock:
                self.lifetime_handshakes += 1
                if ch:
                    self.channel_hs[ch] += 1
                    self._inc_chisto('Handshakes', ch)
            self.last_shake = {'time': time.time(), 'ap': access_point, 'cl': client_station}
        except Exception as e:
            logging.debug(f"[hybrid_tune] on_handshake err: {e}")

    def on_association(self, agent, access_point):
        try:
            ch = access_point.get('channel', 0)
            self._inc_chisto('Associations', ch)
            self._mark_ap_seen(access_point, 'assoc')
            apID = self._ap_id(access_point)
            if apID in self._known_aps:
                self._known_aps[apID]['AT_attacks'] = self._known_aps[apID].get('AT_attacks', 0) + 1
        except Exception as e:
            logging.exception(e)

    def on_deauthentication(self, agent, access_point, client_station):
        try:
            ch = access_point.get('channel', 0)
            self._inc_chisto('Deauths', ch)
            self._mark_ap_seen(access_point, 'deauth')
            apID = self._ap_id(access_point)
            if apID in self._known_aps:
                self._known_aps[apID]['AT_attacks'] = self._known_aps[apID].get('AT_attacks', 0) + 1
        except Exception as e:
            logging.exception(e)

    def on_wifi_update(self, agent, access_points):
        try:
            self._histogram['loops'] += 1

            # mark all known APs as not visible, then re-mark seen ones
            for ap in self._known_aps.values():
                ap['AT_visible'] = False

            active = []
            for ap in access_points:
                self._mark_ap_seen(ap, 'wifi_update')
                ch = ap.get('channel', 0)
                if ch is None or ch < 0:
                    continue
                if ch not in active:
                    active.append(ch)
                    if ch in self._unscanned_channels:
                        self._unscanned_channels.remove(ch)
                    self._dead_channel_counter[ch] = 0
                self._histogram[ch] = self._histogram.get(ch, 0) + 1

            # increment dead counter for channels that had APs before but don't now
            for ch in list(self._dead_channel_counter):
                if ch not in active:
                    self._dead_channel_counter[ch] += 1

            self._active_channels = active
        except Exception as e:
            logging.exception(e)

    def on_bcap_wifi_ap_new(self, agent, event):
        try:
            self._mark_ap_seen(event['data'])
        except Exception as e:
            logging.debug(repr(e))

    def on_bcap_wifi_ap_lost(self, agent, event):
        try:
            ap = event['data']
            apID = self._ap_id(ap)
            channel = ap.get('channel', 0)
            if apID in self._known_aps and self._known_aps[apID].get('AT_visible', False):
                self._known_aps[apID]['AT_visible'] = False
                self._inc_chisto('Current APs', channel, -1)
        except Exception as e:
            logging.debug(repr(e))

    def on_bcap_wifi_client_new(self, agent, event):
        # Clients on a channel = active handshakes possible -> bump priority
        try:
            ap = event.get('data', {}).get('AP', {})
            ch = ap.get('channel', 0)
            if ch:
                self._inc_chisto('Clients', ch)
        except Exception as e:
            logging.debug(repr(e))

    # -------------------- main epoch logic --------------------

    def on_epoch(self, agent, epoch, epoch_data):
        if agent._config.get('ai', {}).get('enabled', False):
            return

        self.ep_data = epoch_data
        self.ep_data['epoch'] = epoch
        diff = None

        try:
            self.epochs_seen += 1
            self.epochs_since_save += 1

            # --- observe ---
            try:
                aps = len(agent.get_access_points())
            except Exception:
                aps = 0
            deauths = int(epoch_data.get('num_deauths', 0) or 0)
            assocs = int(epoch_data.get('num_associations', 0) or 0)
            handshakes = int(epoch_data.get('num_handshakes', 0) or 0)
            missed = int(epoch_data.get('missed_interactions', 0) or 0)
            blind_for = int(epoch_data.get('blind_for_epochs', 0) or 0)
            reward = float(epoch_data.get('reward', 0.0) or 0.0)
            interactions = deauths + assocs
            hs_rate = (handshakes / interactions) if interactions > 0 else 0.0
            missed_rate = (missed / interactions) if interactions > 0 else 0.0

            aps_ema = self._ema_update('aps', aps)
            hs_ema = self._ema_update('hs_rate', hs_rate)
            reward_ema = self._ema_update('reward', reward)
            missed_ema = self._ema_update('missed_rate', missed_rate)

            p = agent._config['personality']
            before = {k: int(p.get(k, 0)) for k in self.BOUNDS}

            # --- track best-ever settings ---
            if self.best_reward is None or reward_ema > self.best_reward + 0.05:
                self.best_reward = reward_ema
                self.best_settings = dict(before)
                logging.info(
                    f"[hybrid_tune] new best reward_ema={reward_ema:.3f} settings={self.best_settings}"
                )

            # --- revert if last adjustment hurt reward ---
            if self.last_adjust and self.reward_before_adjust is not None:
                drop = self.reward_before_adjust - reward_ema
                if drop > self.cfg['reward_drop_threshold']:
                    for k, v in self.last_adjust['from'].items():
                        p[k] = v
                    logging.info(
                        f"[hybrid_tune] REVERT drop={drop:.2f}: "
                        f"rolled back {self.last_adjust['to']} -> {self.last_adjust['from']}"
                    )
                    self.last_adjust = None
                    self.reward_before_adjust = None
                    self._schedule_channels(agent)
                    self._maybe_save()
                    return

            # --- blind panic: crank recon up, loosen rssi ---
            if blind_for >= self.cfg['blind_panic_epochs']:
                p['min_rssi'] = self.BOUNDS['min_rssi'][0]
                p['recon_time'] = self.BOUNDS['recon_time'][1]
                p['hop_recon_time'] = 8
                logging.warning(f"[hybrid_tune] BLIND PANIC blind_for={blind_for}")
                self.last_adjust = None
                self.reward_before_adjust = None
                self._schedule_channels(agent)
                self._maybe_save()
                return

            # --- warmup: just observe ---
            if self.epochs_seen < self.cfg['warmup_epochs']:
                self._schedule_channels(agent)
                self._maybe_save()
                return

            # --- environment-driven tuning ---
            cur = dict(before)

            if aps_ema >= self.cfg['dense_aps']:
                # crowded: tighten RSSI, spend less time on recon
                cur['min_rssi'] = self._clamp('min_rssi', cur['min_rssi'] + 2)
                cur['recon_time'] = self._clamp('recon_time', cur['recon_time'] - 2)
            elif aps_ema <= self.cfg['sparse_aps']:
                # quiet: loosen RSSI, recon longer
                cur['min_rssi'] = self._clamp('min_rssi', cur['min_rssi'] - 2)
                cur['recon_time'] = self._clamp('recon_time', cur['recon_time'] + 2)

            if hs_ema < self.cfg['hs_rate_low'] and interactions >= 3:
                # poor capture: stay longer after deauths, give HS a chance to appear
                cur['hop_recon_time'] = self._clamp('hop_recon_time', cur['hop_recon_time'] + 1)
            elif hs_ema > self.cfg['hs_rate_high']:
                # capture is great: speed up to hit more APs
                cur['hop_recon_time'] = self._clamp('hop_recon_time', cur['hop_recon_time'] - 1)

            if missed_ema > 0.30:
                cur['max_interactions'] = self._clamp('max_interactions', cur['max_interactions'] + 1)
            elif hs_ema > self.cfg['hs_rate_high'] and missed_ema < 0.10:
                cur['max_interactions'] = self._clamp('max_interactions', cur['max_interactions'] - 1)

            cur = self._bias_toward_best(cur)

            diff = {k: (before[k], cur[k]) for k in cur if before[k] != cur[k]}
            if diff:
                for k, v in cur.items():
                    p[k] = v
                self.last_adjust = {'from': before, 'to': cur}
                self.reward_before_adjust = reward_ema
            else:
                self.last_adjust = None
                self.reward_before_adjust = None

            # --- channel scheduling ---
            self._schedule_channels(agent)

            # --- logging ---
            top_ch = sorted(self.channel_hs.items(), key=lambda x: -x[1])[:5]
            top_ch_str = ', '.join(f'{c}:{n}' for c, n in top_ch) or 'none'
            tag = f'adj={diff}' if diff else 'stable'
            logging.info(
                f"[hybrid_tune] aps_ema={aps_ema:.1f} hs_ema={hs_ema:.2f} "
                f"missed_ema={missed_ema:.2f} reward_ema={reward_ema:.2f} "
                f"lifetime_hs={self.lifetime_handshakes} captured={len(self._captured_aps)} "
                f"top_ch={top_ch_str} {tag}"
            )

            self._maybe_save()
        except Exception as e:
            logging.exception(f"[hybrid_tune] epoch error: {e}")

    def _schedule_channels(self, agent):
        """Pick next scan channels: active + extras (priority-weighted + explore)."""
        try:
            next_channels = list(self._active_channels)
            n = int(self.cfg.get('extra_channels', 3))

            # repopulate unscanned pool if empty
            if not self._unscanned_channels:
                if "restrict_channels" in self.options:
                    self._unscanned_channels = list(self.options["restrict_channels"])
                elif hasattr(agent, "_allowed_channels"):
                    self._unscanned_channels = list(agent._allowed_channels)
                elif hasattr(agent, "_supported_channels"):
                    self._unscanned_channels = list(agent._supported_channels)
                else:
                    self._unscanned_channels = pwnagotchi.utils.iface_channels(
                        agent._config['main']['iface']
                    )

            # filter out channels in cooldown
            cooldown = int(self.cfg.get('dead_channel_cooldown', 5))
            pool = [c for c in self._unscanned_channels
                    if self._dead_channel_counter.get(c, 0) < cooldown]
            if not pool:  # all dead? reset so we re-explore
                self._dead_channel_counter.clear()
                pool = list(self._unscanned_channels)

            # split picks: priority (weighted by HS history) + exploration (random)
            priority_weight = float(self.cfg.get('priority_channel_weight', 0.70))
            n_priority = max(1, int(round(n * priority_weight))) if n > 0 else 0
            n_explore = max(0, n - n_priority)

            priority_pool = [c for c in pool if self.channel_hs.get(c, 0) > 0
                             or self._chistos.get('_all_actions', {}).get(c, 0) > 0]
            priority_picks = self._pick_weighted(priority_pool, n_priority)

            shortfall = n_priority - len(priority_picks)
            explore_pool = [c for c in pool if c not in priority_picks]
            explore_target = min(n_explore + shortfall, len(explore_pool))
            explore_picks = random.sample(explore_pool, explore_target) if explore_pool else []

            added = priority_picks + explore_picks
            for ch in added:
                if ch in self._unscanned_channels:
                    self._unscanned_channels.remove(ch)
                next_channels.append(ch)

            agent._config['personality']['channels'] = next_channels
            logging.info(
                f"[hybrid_tune] active={self._active_channels} "
                f"priority={priority_picks} explore={explore_picks} "
                f"unscanned={len(self._unscanned_channels)}"
            )
        except Exception as e:
            logging.exception(f"[hybrid_tune] schedule err: {e}")

    # -------------------- UI --------------------

    def on_ui_setup(self, ui):
        self._ui = ui
        self._orig_mode = ui.get('mode')

    def on_ui_update(self, ui):
        # leave pwnagotchi's default display alone to avoid clashing with Fancygotchi
        pass

    def on_unload(self, ui):
        self._save_state()

    # -------------------- web UI --------------------

    def on_webhook(self, path, request):
        if not self._agent:
            return render_template_string(
                "<html><body><h1>HybridTune not ready yet</h1></body></html>"
            )
        try:
            if request.method == "GET" and (path == "/" or not path):
                ret = '<html><head><title>HybridTune</title>'
                ret += '<style>body{font-family:monospace;}table{border-collapse:collapse;}'
                ret += 'th,td{border:1px solid #888;padding:4px 8px;}'
                ret += 'th{background:#222;color:#eee;}</style></head><body>'
                ret += '<h1>HybridTune v%s</h1>' % self.__version__
                ret += self._html_status()
                ret += self._html_chistos()
                ret += self._html_top_aps()
                ret += '</body></html>'
                return render_template_string(ret)
        except Exception as e:
            logging.exception(f"[hybrid_tune] webhook: {e}")
            return render_template_string(
                f"<html><body><h1>Error</h1><pre>{html.escape(repr(e))}</pre></body></html>"
            )
        return render_template_string("<html><body><h1>Not found</h1></body></html>")

    @staticmethod
    def _fmt(v, spec='.3f'):
        if v is None:
            return 'N/A'
        try:
            return format(v, spec)
        except Exception:
            return str(v)

    def _html_status(self):
        ret = '<h2>Status</h2><table>'
        ret += f'<tr><td>Epochs seen</td><td>{self.epochs_seen}</td></tr>'
        ret += f'<tr><td>Lifetime handshakes</td><td>{self.lifetime_handshakes}</td></tr>'
        ret += f'<tr><td>Unique pwns this session</td><td>{len(self._captured_aps)}</td></tr>'
        ret += f'<tr><td>Known APs</td><td>{len(self._known_aps)}</td></tr>'
        for k, v in self.ema.items():
            ret += f'<tr><td>EMA {html.escape(k)}</td><td>{self._fmt(v)}</td></tr>'
        ret += f'<tr><td>Best reward</td><td>{self._fmt(self.best_reward)}</td></tr>'
        if self.best_settings:
            ret += f'<tr><td>Best settings</td><td>{html.escape(str(self.best_settings))}</td></tr>'
        p = self._agent._config.get('personality', {}) if self._agent else {}
        cur = {k: p.get(k, '?') for k in self.BOUNDS}
        ret += f'<tr><td>Current personality</td><td>{html.escape(str(cur))}</td></tr>'
        ret += '</table>'
        return ret

    def _html_chistos(self):
        ret = '<h2>Channel stats (this session)</h2>'
        all_actions = self._chistos.get('_all_actions', {})
        channels = sorted([c for c in all_actions if c != -1],
                          key=lambda c: -all_actions[c])[:25]
        if not channels:
            return ret + '<p>No data collected yet.</p>'

        stats = [s for s in self._chistos if s != '_all_actions']
        ret += '<table><tr><th>Channel</th>'
        for s in stats:
            ret += f'<th>{html.escape(s)}</th>'
        ret += '<th>Lifetime HS</th></tr>'

        for ch in channels:
            ret += f'<tr><td>{ch}</td>'
            for s in stats:
                v = self._chistos[s].get(ch, 0)
                ret += f'<td align=right>{v}</td>'
            ret += f'<td align=right>{self.channel_hs.get(ch, 0)}</td>'
            ret += '</tr>'
        ret += '</table>'
        return ret

    def _html_top_aps(self):
        ret = '<h2>Top APs by handshakes (this session)</h2>'
        hs_aps = [(i, a) for i, a in self._known_aps.items() if a.get('AT_handshake', 0) > 0]
        if not hs_aps:
            return ret + '<p>No handshakes yet.</p>'
        hs_aps.sort(key=lambda x: -x[1]['AT_handshake'])

        ret += '<table><tr><th>Name</th><th>MAC</th><th>Ch</th><th>RSSI</th>'
        ret += '<th>HS</th><th>Attacks</th><th>Seen</th></tr>'
        for _id, ap in hs_aps[:40]:
            ret += f'<tr>'
            ret += f'<td>{html.escape(str(ap.get("hostname", "?")))}</td>'
            ret += f'<td>{html.escape(str(ap.get("mac", "?")))}</td>'
            ret += f'<td>{ap.get("channel", "?")}</td>'
            ret += f'<td>{ap.get("rssi", "?")}</td>'
            ret += f'<td>{ap.get("AT_handshake", 0)}</td>'
            ret += f'<td>{ap.get("AT_attacks", 0)}</td>'
            ret += f'<td>{ap.get("AT_seen", 0)}</td>'
            ret += '</tr>'
        ret += '</table>'
        return ret
