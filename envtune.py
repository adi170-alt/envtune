import json
import logging
import os
import threading
import time
from collections import defaultdict

import pwnagotchi.plugins as plugins


class EnvTune(plugins.Plugin):
    __author__ = 'adi1708'
    __version__ = '3.0.0'
    __license__ = 'MIT'
    __description__ = 'Environment-aware personality tuner with EMA smoothing, persistence, blind-panic, reward-revert, and best-settings memory.'

    STATE_PATH = '/etc/pwnagotchi/envtune_state.json'

    BOUNDS = {
        'min_rssi': (-85, -65),
        'hop_recon_time': (4, 15),
        'recon_time': (15, 45),
        'max_interactions': (2, 6),
    }

    DEFAULTS = {
        'ema_alpha': 0.35,
        'warmup_epochs': 3,
        'hs_rate_low': 0.15,
        'hs_rate_high': 0.40,
        'dense_aps': 25,
        'sparse_aps': 8,
        'blind_panic_epochs': 3,
        'reward_drop_threshold': 0.25,
        'save_every_n_epochs': 5,
        'best_bias_weight': 0.15,
    }

    def __init__(self):
        self.cfg = dict(self.DEFAULTS)
        self.ema = {'aps': None, 'hs_rate': None, 'reward': None, 'missed_rate': None}
        self.epochs_seen = 0
        self.epochs_since_save = 0
        self.session_start = time.time()
        self.lifetime_handshakes = 0
        self.channel_hs = defaultdict(int)
        self.last_adjust = None
        self.reward_before_adjust = None
        self.best_reward = None
        self.best_settings = None
        self._lock = threading.Lock()

    def on_loaded(self):
        try:
            user = self.options or {}
            self.cfg.update({k: v for k, v in user.items() if k in self.DEFAULTS})
        except Exception:
            pass
        self._load_state()
        logging.info(
            f"[envtune] v3 loaded alpha={self.cfg['ema_alpha']} warmup={self.cfg['warmup_epochs']} "
            f"lifetime_hs={self.lifetime_handshakes} best_reward={self.best_reward}"
        )

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
            logging.info(f"[envtune] state loaded, lifetime_hs={self.lifetime_handshakes}")
        except Exception as e:
            logging.warning(f"[envtune] state load failed: {e}")

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
            logging.warning(f"[envtune] state save failed: {e}")

    def _ema_update(self, key, value):
        a = self.cfg['ema_alpha']
        prev = self.ema.get(key)
        if prev is None or (prev == 0 and self.epochs_seen < self.cfg['warmup_epochs']):
            new = float(value)
        else:
            new = a * float(value) + (1 - a) * prev
        self.ema[key] = new
        return new

    def _clamp(self, key, v):
        lo, hi = self.BOUNDS[key]
        return max(lo, min(hi, int(round(v))))

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

    def on_handshake(self, agent, filename, access_point, client_station):
        try:
            ch = 0
            if isinstance(access_point, dict):
                ch = int(access_point.get('channel', 0) or 0)
            with self._lock:
                self.lifetime_handshakes += 1
                if ch:
                    self.channel_hs[ch] += 1
        except Exception as e:
            logging.debug(f"[envtune] on_handshake err: {e}")

    def on_epoch(self, agent, epoch, epoch_data):
        try:
            self.epochs_seen += 1
            self.epochs_since_save += 1

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

            if self.best_reward is None or reward_ema > self.best_reward + 0.05:
                self.best_reward = reward_ema
                self.best_settings = dict(before)
                logging.info(f"[envtune] new best reward_ema={reward_ema:.3f} settings={self.best_settings}")

            if self.last_adjust and self.reward_before_adjust is not None:
                drop = self.reward_before_adjust - reward_ema
                if drop > self.cfg['reward_drop_threshold']:
                    for k, v in self.last_adjust['from'].items():
                        p[k] = v
                    logging.info(
                        f"[envtune] REVERT drop={drop:.2f}: rolled back {self.last_adjust['to']} -> {self.last_adjust['from']}"
                    )
                    self.last_adjust = None
                    self.reward_before_adjust = None
                    self._maybe_save()
                    return

            if blind_for >= self.cfg['blind_panic_epochs']:
                p['min_rssi'] = self.BOUNDS['min_rssi'][0]
                p['recon_time'] = self.BOUNDS['recon_time'][1]
                p['hop_recon_time'] = 8
                logging.warning(f"[envtune] BLIND PANIC blind_for={blind_for} -> safe-permissive preset")
                self.last_adjust = None
                self.reward_before_adjust = None
                self._maybe_save()
                return

            if self.epochs_seen < self.cfg['warmup_epochs']:
                self._maybe_save()
                return

            cur = dict(before)

            if aps_ema >= self.cfg['dense_aps']:
                cur['min_rssi'] = self._clamp('min_rssi', cur['min_rssi'] + 2)
                cur['recon_time'] = self._clamp('recon_time', cur['recon_time'] - 2)
            elif aps_ema <= self.cfg['sparse_aps']:
                cur['min_rssi'] = self._clamp('min_rssi', cur['min_rssi'] - 2)
                cur['recon_time'] = self._clamp('recon_time', cur['recon_time'] + 2)

            if hs_ema < self.cfg['hs_rate_low'] and interactions >= 3:
                cur['hop_recon_time'] = self._clamp('hop_recon_time', cur['hop_recon_time'] + 1)
            elif hs_ema > self.cfg['hs_rate_high']:
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

            elapsed_h = max(0.01, (time.time() - self.session_start) / 3600.0)
            hs_per_hour = self.lifetime_handshakes / elapsed_h if self.epochs_seen < 30 else handshakes / (epoch_data.get('duration_secs', 60) / 3600.0) if epoch_data.get('duration_secs', 0) > 0 else 0
            top_ch = sorted(self.channel_hs.items(), key=lambda x: -x[1])[:5]
            top_ch_str = ', '.join(f'{c}:{n}' for c, n in top_ch) or 'none'

            tag = f'adj={diff}' if diff else 'stable'
            logging.info(
                f"[envtune] aps_ema={aps_ema:.1f} hs_rate_ema={hs_ema:.2f} missed_ema={missed_ema:.2f} "
                f"reward_ema={reward_ema:.2f} lifetime_hs={self.lifetime_handshakes} top_ch={top_ch_str} {tag}"
            )

            self._maybe_save()
        except Exception as e:
            logging.exception(f"[envtune] epoch error: {e}")

    def _maybe_save(self):
        if self.epochs_since_save >= self.cfg['save_every_n_epochs']:
            self._save_state()
            self.epochs_since_save = 0
