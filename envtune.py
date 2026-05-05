#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
envtune.py  —  Adaptive Environment Tuner for Pwnagotchi
=========================================================
Version   : 1.2.0
License   : MIT
Repository: https://github.com/adi170-alt/envtune

A drop-in replacement for the removed pwnagotchi AI, built specifically
for jayofelony/pwnagotchi (noai branch). Uses Sliding-Window UCB1 — a
proven reinforcement learning technique — with a contextual state space
extended by GPS zones, thermal safety, client awareness, and smart
channel scheduling. Gets measurably better every session.

Why this exists
───────────────
Jay removed the A2C neural network because it destabilised the wifi
firmware and drained batteries. The stock pwnagotchi is still strong,
but runs on fixed parameters — it can't adapt to your specific routes,
times, or environments. EnvTune fills that gap using lightweight ML
(≈ 2-3% CPU on a Pi Zero 2 W) that cannot crash the radio.

What it learns per environmental context
─────────────────────────────────────────
12 personality parameters across 108 contexts (density × time × trend ×
mobility). UCB1 with a sliding window means the learning adapts as
environments change — a stale memory of "what worked in 2024" does not
override fresh evidence of "what works here now".

Key capabilities
────────────────
 • 12-parameter UCB learning (ALL verified against jayofelony defaults)
 • Hierarchical priors so rare contexts benefit from common ones
 • Proper bettercap sync for wifi.ap.ttl / sta.ttl / min.rssi
 • GPS zone-aware learning (optional, auto-detects TheyLive/stock gps)
 • Stationary vs mobile detection via speed/AP-turnover
 • Heatmap of captures with zone productivity scoring
 • Thermal safety (Pi can crash >80°C — we back off at 70°C)
 • Client-aware targeting (deauth needs clients; PMKID doesn't)
 • PMF detection (stops wasting deauths on protected networks)
 • Per-AP cooldown on persistent non-responders
 • Already-captured detection from /root/handshakes/
 • Free-channel opportunism via on_free_channel callback
 • PiSugar battery awareness (optional, graceful)
 • wpa-sec cracked-feedback loop (if potfile exists)
 • Whitelist respect (doesn't skew learning on skipped APs)
 • Nexmon crash detection + automatic backoff
 • Async state save (no SD-card IO stalls mid-epoch)
 • Version-migrated state (survives plugin upgrades)
 • Full web UI at /plugins/envtune/ with explanatory tooltips
 • Five CPU profiles: minimal, light, balanced, aggressive, beast

Requirements
────────────
 • jayofelony/pwnagotchi (noai branch)     — verified compatible
 • Python 3.7+                             — part of stock image
 • No extra pip packages                   — uses only stdlib + flask

Installation
────────────
 1) Copy this file to /usr/local/share/pwnagotchi/custom-plugins/envtune.py
 2) In /etc/pwnagotchi/config.toml add:

        main.plugins.envtune.enabled = true

 3) (Optional) pick a CPU profile for your hardware:

        main.plugins.envtune.cpu_profile = "balanced"
        # choices: "minimal" "light" "balanced" "aggressive" "beast"
        # default: auto-detects based on /proc/cpuinfo

 4) (Optional) GPS integration — works automatically if TheyLive or
    stock gps plugin is enabled. No config required.

 5) (Optional) PiSugar — automatically detected if pisugarx is enabled.

 6) (Optional) Turn off stock auto_tune if you use it — envtune replaces
    it completely.

 7) Reboot. First 20 epochs = warmup + exploration. After ~200 epochs
    the plugin begins consistently choosing optimal parameters for each
    state you encounter. It gets smarter forever.

Config presets (add to config.toml to override defaults)
─────────────────────────────────────────────────────────
For aggressive wardriving:
    main.plugins.envtune.cpu_profile = "aggressive"
    main.plugins.envtune.temp_critical = 80.0
    main.plugins.envtune.extra_channels = 5

For stealthy home use:
    main.plugins.envtune.cpu_profile = "light"
    main.plugins.envtune.ucb_c = 1.0
    main.plugins.envtune.opportunistic_overrides = false

For max learning on strong hardware (Pi 4/5):
    main.plugins.envtune.cpu_profile = "beast"
    main.plugins.envtune.ucb_window = 80
    main.plugins.envtune.save_every_n_epochs = 10

Web UI
──────
http://<pwnagotchi>:8080/plugins/envtune/
Shows live stats, UCB learning table, channel productivity, AP
intelligence, GPS zones, and thermal status. Hover any value for an
explanation.

Credits
───────
Built on top of prior art by:
  • @evilsocket   — original pwnagotchi
  • @jayofelony   — noai fork
  • @Sniffleupagus — auto_tune plugin
  • @rai68 + @AlienMajik — TheyLive GPS plugin
  • @adi1708(⌐■_■)        — earlier envtune iterations
"""

import hmac
import html
import json
import logging
import math
import os
import queue
import random
import secrets
import tempfile
import threading
import time
from collections import defaultdict, deque

import pwnagotchi.plugins as plugins
import pwnagotchi.utils
from flask import make_response


# ═══════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════

def _si(v, default=0):
    """Safe int cast — never raises."""
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _sf(v, default=0.0):
    """Safe float cast — never raises."""
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two (lat, lon) pairs."""
    R = 6371000.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _is_valid_mac(mac):
    """Validate a MAC string of form aa:bb:cc:dd:ee:ff."""
    if not mac or not isinstance(mac, str):
        return False
    parts = mac.split(':')
    if len(parts) != 6:
        return False
    for p in parts:
        if len(p) != 2 or not all(c in '0123456789abcdefABCDEF' for c in p):
            return False
    return True


def _format_mac_colons(mac_n):
    """Convert normalised 12-char MAC to colon-separated form. Returns '' on bad input."""
    if not mac_n or len(mac_n) != 12:
        return ''
    return ':'.join(mac_n[i:i+2] for i in range(0, 12, 2))


def _detect_hardware():
    """
    Detect Pi hardware for CPU profile defaults.
    Returns one of: 'pi_zero', 'pi_zero_2', 'pi_3', 'pi_4', 'pi_5', 'unknown'
    """
    try:
        with open('/proc/cpuinfo') as f:
            info = f.read().lower()
        if 'pi zero 2' in info or 'bcm2710' in info or 'cortex-a53' in info:
            return 'pi_zero_2'
        if 'pi zero' in info or 'bcm2835' in info:
            return 'pi_zero'
        if 'bcm2837' in info or 'pi 3' in info:
            return 'pi_3'
        if 'bcm2711' in info or 'pi 4' in info:
            return 'pi_4'
        if 'bcm2712' in info or 'pi 5' in info:
            return 'pi_5'
    except Exception:
        pass
    return 'unknown'


# CPU profile definitions — balance between learning quality and CPU
#
#   ucb_window          — how many recent rewards UCB remembers per arm
#   zone_resolution_m   — GPS zone cell size (smaller = more zones = harder to learn)
#   save_every_n        — how often to persist state to SD card
#   ap_track_max        — hard cap on AP dict size (memory control)
#   extra_channels      — how many non-active channels to include per hop
#   ucb_cache_epochs    — cache UCB selections for N epochs (saves math)
#   enable_proactive    — permit opportunistic wifi.assoc injection
CPU_PROFILES = {
    'minimal': {
        'ucb_window': 20, 'zone_resolution_m': 300, 'save_every_n': 15,
        'ap_track_max': 150, 'extra_channels': 2, 'ucb_cache_epochs': 3,
        'enable_proactive': False,
    },
    'light': {
        'ucb_window': 30, 'zone_resolution_m': 200, 'save_every_n': 10,
        'ap_track_max': 250, 'extra_channels': 3, 'ucb_cache_epochs': 2,
        'enable_proactive': False,
    },
    'balanced': {
        'ucb_window': 40, 'zone_resolution_m': 150, 'save_every_n': 5,
        'ap_track_max': 400, 'extra_channels': 3, 'ucb_cache_epochs': 1,
        'enable_proactive': True,
    },
    'aggressive': {
        'ucb_window': 60, 'zone_resolution_m': 100, 'save_every_n': 5,
        'ap_track_max': 600, 'extra_channels': 4, 'ucb_cache_epochs': 1,
        'enable_proactive': True,
    },
    'beast': {
        'ucb_window': 80, 'zone_resolution_m': 75, 'save_every_n': 5,
        'ap_track_max': 1000, 'extra_channels': 5, 'ucb_cache_epochs': 0,
        'enable_proactive': True,
    },
}

# Hardware → default profile
HW_DEFAULT_PROFILE = {
    'pi_zero':   'light',
    'pi_zero_2': 'balanced',    # that's you — Pi Zero 2 W overclocked
    'pi_3':      'balanced',
    'pi_4':      'aggressive',
    'pi_5':      'beast',
    'unknown':   'balanced',
}


# ═══════════════════════════════════════════════════════════════════════════
# Main Plugin class
# ═══════════════════════════════════════════════════════════════════════════

class EnvTune(plugins.Plugin):
    __author__      = 'adi1708'
    __version__     = '1.2.0'
    __license__     = 'MIT'
    __description__ = ('Adaptive environment tuner — drop-in replacement '
                       'for the removed pwnagotchi AI. Learns optimal '
                       'parameters per context using Sliding-Window UCB1, '
                       'with GPS zones, thermal safety, and smart channel '
                       'scheduling. Maximises unique handshake captures.')

    # ── Paths ─────────────────────────────────────────────────────────────
    STATE_PATH    = '/etc/pwnagotchi/envtune_state.json'
    HANDSHAKE_DIR = '/root/handshakes'
    GPS_TRACK     = '/root/pwnagotchi_gps_track.ndjson'   # TheyLive
    WPASEC_POT    = '/root/handshakes/wpa-sec.cracked.potfile'

    # State schema — bumped on breaking changes, migrate on load
    STATE_SCHEMA_VERSION = 3

    # ── UCB arms — VERIFIED against jayofelony defaults.toml (noai) ───────
    # Every single parameter here actually exists and affects pwnagotchi
    # behaviour. No fake parameters this time.
    UCB_ARMS = {
        # Core attack tuning
        'min_rssi':                  [-85, -80, -75, -70, -65],
        'hop_recon_time':            [4, 6, 8, 10, 12, 15],
        'min_recon_time':            [2, 3, 5, 7, 10],
        'recon_time':                [15, 20, 25, 30, 35, 45],
        'max_interactions':          [2, 3, 4, 5, 6],

        # AP/client retention in bettercap
        'ap_ttl':                    [60, 120, 180, 300, 600],
        'sta_ttl':                   [120, 300, 600, 900],

        # Recon dynamics (how pwnagotchi reacts to inactivity)
        'max_misses_for_recon':      [3, 5, 7, 10],
        'max_inactive_scale':        [2, 3, 5],
        'recon_inactive_multiplier': [1, 2, 3],

        # Jay's throttles (radio pause between attacks — float seconds)
        'throttle_a':                [0.2, 0.4, 0.6, 0.8, 1.0],
        'throttle_d':                [0.3, 0.5, 0.7, 0.9, 1.2],

        # Mood thresholds (how many epochs before pwnagotchi flips emotion).
        # These influence its decision to take a break / change mode and
        # therefore indirectly affect handshake productivity. The original
        # AI tuned these too — they are safe to learn (no firmware impact).
        'bored_num_epochs':          [10, 15, 20, 25],
        'sad_num_epochs':            [15, 20, 25, 30],
    }

    # Hard bounds for safety clamping during panic/thermal modes
    BOUNDS = {
        'min_rssi':                  (-85,  -65),
        'hop_recon_time':            (4,     15),
        'min_recon_time':            (2,     10),
        'recon_time':                (15,    45),
        'max_interactions':          (2,      6),
        'ap_ttl':                    (60,   600),
        'sta_ttl':                   (120,  900),
        'max_misses_for_recon':      (3,     10),
        'max_inactive_scale':        (2,      5),
        'recon_inactive_multiplier': (1,      3),
        'throttle_a':                (0.2,  1.0),
        'throttle_d':                (0.3,  1.2),
        'bored_num_epochs':          (10,    30),
        'sad_num_epochs':            (15,    30),
    }

    # Parameters that need explicit bettercap sync (wifi.* namespace).
    # Bettercap since commit 12a11ef applies these in realtime when
    # changed via "set" command. Writing the dict is NOT enough.
    BETTERCAP_SYNC_MAP = {
        'min_rssi': 'wifi.rssi.min',
        'ap_ttl':   'wifi.ap.ttl',
        'sta_ttl':  'wifi.sta.ttl',
    }

    # Non-overlapping channels get a scoring bonus (less interference)
    NON_OVERLAPPING = {1, 6, 11, 36, 40, 44, 48, 149, 153, 157, 161}

    # Mobility categories based on speed (m/s)
    MOBILITY_STATIONARY = 'stationary'  # < 0.5 m/s (≈ 1.8 km/h)
    MOBILITY_WALKING    = 'walking'     # 0.5 – 3.0 m/s (walk/jog)
    MOBILITY_MOBILE     = 'mobile'      # > 3.0 m/s (bike/car)

    # ── Default config (all overridable via main.plugins.envtune.*) ───────
    DEFAULTS = {
        # Core
        'cpu_profile':               None,     # auto-detect if None
        'ema_alpha':                 0.30,
        'warmup_epochs':             5,
        'dense_aps':                 25,
        'sparse_aps':                8,

        # UCB
        'ucb_c':                     1.4,
        'reward_delay':              3,
        'ucb_c_floor':               0.6,      # lowest C may decay to
        'ucb_c_anneal_epochs':       500,
        # Empirical-Bayes shrinkage strength (ANNEALED): with n local
        # samples per arm we trust the local mean by n/(n+k). Cold-start
        # we want HEAVY shrinkage (k_max≈5: at n=5 you're 50/50 with the
        # parent). Late-game we want LIGHT shrinkage (k_min≈1: at n=5
        # you're 83% local) so genuinely-better-but-undertested arms
        # aren't pulled back to the mediocre prior forever.
        # k decays linearly with total real samples observed, capped at
        # ucb_shrinkage_anneal_samples. Telemetry showed v1.1's fixed
        # k=5 was holding back arms like min_rssi=-75 (mean=0.40, n=13)
        # whose effective mean was being pulled down to ~0.37.
        'ucb_shrinkage_k_max':       5.0,
        'ucb_shrinkage_k_min':       1.0,
        'ucb_shrinkage_anneal_samples': 500,
        # Back-compat key — if user's config sets `ucb_shrinkage_k`
        # explicitly we keep using that as a fixed override (no anneal).
        'ucb_shrinkage_k':           None,

        # Stagnation / exploration
        'stagnation_epochs':         12,
        'exploration_boost_c':       2.5,
        'exploration_boost_epochs':  6,

        # Blind panic
        'blind_panic_epochs':        3,
        'blind_recovery_steps':      5,

        # AP targeting
        'ap_cooldown_attacks':       12,
        'ap_cooldown_short':         15,
        'ap_cooldown_long':          50,
        'pmf_attack_threshold':      10,
        'client_recency_epochs':     3,
        'missed_cooldown_threshold': 5,  # AP marked missed this many times → cooldown

        # Channel scheduling
        'priority_channel_weight':   0.70,
        'dead_channel_cooldown':     5,
        'dead_ch_lifetime_weight':   0.01,

        # Thermal safety
        'temp_warn':                 70.0,
        'temp_critical':             78.0,

        # Misc
        'reset_history':             True,
        'opportunistic_min_gap':     2,
        'opportunistic_overrides':   True,

        # GPS
        'enable_gps':                True,     # auto-disables if no GPS
        'gps_stale_seconds':         90,
        'mobility_walk_threshold':   0.5,
        'mobility_mobile_threshold': 3.0,

        # Proactive attacks (opt-in even on profiles that allow it)
        'proactive_min_rssi':        -68,
        'proactive_min_clients':     3,
        'proactive_gap_epochs':      5,

        # Battery (pisugar integration)
        'battery_low_threshold':     20.0,
        'battery_critical_threshold': 10.0,

        # wpa-sec feedback
        'enable_wpasec_feedback':    True,
        'potfile_rescan_every_n':    100,    # epochs
        'handshake_rescan_every_n':  200,    # epochs

        # Logging
        'log_level':                 'INFO',
    }

    # ─────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────

    def __init__(self):
        # Config — finalised in on_loaded after options merge
        self.cfg = dict(self.DEFAULTS)
        self._profile = None          # populated in on_loaded

        # EMA-smoothed observation signals
        self.ema = {k: None for k in (
            'aps', 'hs_rate', 'reward', 'missed_rate', 'hs_per_min',
            'active_ratio', 'inactive_ratio', 'hops_per_epoch',
            'temperature', 'cpu_load', 'speed',
        )}
        self._prev_reward_ema = None
        self._reward_trend    = 0.0
        self._last_reward_breakdown = {}
        self._prev_aps_ema    = None   # persists between epochs for crash detect

        # Adaptive reward target — learns what "good hs/min" means here
        self._recent_hpm      = deque(maxlen=60)
        self._reward_history  = deque(maxlen=60)   # for rolling-median stagnation

        # Epoch counters / state machine
        self.epochs_seen          = 0
        self.epochs_since_save    = 0
        self._stagnation_count    = 0
        self._exploration_boost   = 0
        self._blind_recovery      = 0
        self._blind_saved_params  = None
        self._crash_suspect       = 0
        self._last_override_ep    = -99
        self._last_proactive_ep   = -99
        self._thermal_throttle    = False
        self._mood                = 'neutral'
        self._battery_level       = None    # pisugar integration
        self._session_hs_bssids   = set()   # for per-epoch new_unique calc
        self._lifetime_new_count  = 0        # cumulative count of LIFETIME-new captures

        # UCB — initialised in on_loaded
        self.ucb_table        = {}
        self._decision_buffer = deque(maxlen=5)
        self._ucb_cache       = {}    # (param,state) -> (arm, epoch_set)
        self._ucb_cache_epoch = -1

        # Which pwnagotchi params this fork actually exposes
        self._active_params = set(self.UCB_ARMS.keys())

        # Best-reward tracking (telemetry)
        self.best_reward   = None
        self.best_settings = None

        # Lifetime stats (persistent)
        self.lifetime_handshakes = 0
        self.session_start_wall  = time.time()
        self.session_start_mono  = time.monotonic()

        # Counters synced after _load_state in on_loaded (prevents inflated
        # diff on first epoch when state has lifetime_new_count > 0).
        self._lifetime_new_count_prev = 0
        self._known_aps_count_prev    = 0
        self._last_loc_change_ep      = -99

        # Bettercap dynamic skip-list — captured BSSIDs we ask bettercap to
        # deprioritise so radio time goes to *new* targets (the whole point).
        self._bcap_skip_macs           = set()
        self._bcap_skip_pushed_count   = 0

        # Channel lifetime dict  ch → stats (persistent)
        self._ch_lt = defaultdict(lambda: {
            'hs': 0, 'assocs': 0, 'deauths': 0,
            'clients': 0, 'visits': 0, 'wasted': 0,
            'free_seen': 0, 'passive_hs': 0, 'cracked': 0,
        })
        self._dead_lt = defaultdict(int)

        # Session-only channel state
        self._chistos            = {'_all_actions': {-1: 0}}
        self._active_channels    = []
        self._unscanned_channels = []
        self._dead_session       = defaultdict(int)
        self._free_channels      = deque(maxlen=8)   # recent free-channel reports

        # AP tracking
        self._known_aps        = {}
        self._captured_aps     = set()     # apIDs with HS this session
        self._captured_bssids  = set()     # BSSIDs seen in /root/handshakes/
        self._cracked_bssids   = set()     # BSSIDs with known password (wpa-sec)
        self._whitelist_macs   = set()
        self._whitelist_ssids  = set()

        # GPS
        self._gps_available   = False
        self._gps_source      = None       # 'theylive', 'stock_gps', or None
        self._gps_last_fix    = None       # {'lat', 'lon', 'speed', 'ts_mono'}
        self._gps_zones       = defaultdict(lambda: {
            'hs': 0, 'attacks': 0, 'visits': 0,
            'last_seen': 0.0, 'channels': defaultdict(int),
        })
        self._current_zone    = None
        self._current_mobility = self.MOBILITY_STATIONARY

        # Location-change detection (works with or without GPS)
        self._loc_fp_stored = None
        self._fp_history    = deque(maxlen=12)

        # Thread safety
        self._state_lock = threading.RLock()

        # Async save thread
        self._save_queue   = queue.Queue(maxsize=4)
        self._save_thread  = None
        self._save_stop    = threading.Event()

        # Web UI CSRF token — bound to the running process. POST endpoints
        # require this token to prevent cross-site request forgery from a
        # browser visiting an attacker page on the same LAN.
        self._csrf_token = secrets.token_urlsafe(24)
        self._action_log = deque(maxlen=20)

        # Plugin wiring
        self._agent     = None
        self._ui        = None
        self.last_shake = {'time': time.time()}

    # ─────────────────────────────────────────────────────────────────────
    # State-space definition
    # ─────────────────────────────────────────────────────────────────────
    # State key format: "density_tod_trend_mobility"
    #   density:  sparse / medium / dense
    #   tod:      night / morning / afternoon / evening
    #   trend:    falling / stable / rising
    #   mobility: stationary / walking / mobile
    # Total: 3×4×3×3 = 108 base states (plus optional GPS zone suffix)
    # ─────────────────────────────────────────────────────────────────────

    def _all_states(self):
        return [
            f'{d}_{t}_{r}_{m}'
            for d in ('sparse', 'medium', 'dense')
            for t in ('night', 'morning', 'afternoon', 'evening')
            for r in ('falling', 'stable', 'rising')
            for m in (self.MOBILITY_STATIONARY, self.MOBILITY_WALKING,
                      self.MOBILITY_MOBILE)
        ]

    def _init_ucb_table(self):
        """Build empty UCB tables for all (param, state, arm) triples."""
        W      = int(self._profile['ucb_window'])
        states = self._all_states()
        self.ucb_table = {
            param: {
                state: {arm: {'n': 0, 'rewards': deque(maxlen=W)} for arm in arms}
                for state in states
            }
            for param, arms in self.UCB_ARMS.items()
        }

    def _ensure_state(self, param, state):
        """Lazy-create UCB entry (handles new states after version bumps)."""
        W = int(self._profile['ucb_window'])
        if param not in self.ucb_table:
            self.ucb_table[param] = {}
        if state not in self.ucb_table[param]:
            self.ucb_table[param][state] = {
                arm: {'n': 0, 'rewards': deque(maxlen=W)}
                for arm in self.UCB_ARMS[param]
            }

    # ─────────────────────────────────────────────────────────────────────
    # UCB serialisation (with version migration)
    # ─────────────────────────────────────────────────────────────────────

    def _serialise_ucb(self):
        out = {}
        for param, states in self.ucb_table.items():
            out[param] = {}
            for state, arms in states.items():
                out[param][state] = {
                    str(arm): {'n': d['n'], 'rewards': list(d['rewards'])}
                    for arm, d in arms.items()
                }
        return out

    def _deserialise_ucb(self, raw, loaded_schema):
        """Load UCB table with on-the-fly state-key migration."""
        W = int(self._profile['ucb_window'])
        for param, states in raw.items():
            if param not in self.ucb_table:
                continue  # param removed in newer version — skip gracefully
            for old_state, arms in states.items():
                # Migration: add missing mobility suffix to old states
                new_state = self._migrate_state_key(old_state, loaded_schema)
                self._ensure_state(param, new_state)
                for arm_s, d in arms.items():
                    try:
                        ref_type = type(self.UCB_ARMS[param][0])
                        arm      = ref_type(arm_s)
                    except (ValueError, TypeError):
                        try:
                            arm = float(arm_s)
                        except (ValueError, TypeError):
                            continue
                    if arm in self.ucb_table[param][new_state]:
                        entry = self.ucb_table[param][new_state][arm]
                        entry['n'] = _si(d.get('n', 0))
                        rews = d.get('rewards', []) or []
                        # If migrating, merge rather than overwrite
                        if len(entry['rewards']) > 0:
                            combined = list(entry['rewards']) + rews
                            entry['rewards'] = deque(combined[-W:], maxlen=W)
                        else:
                            entry['rewards'] = deque(rews, maxlen=W)

    def _migrate_state_key(self, old_key, from_schema):
        """
        Migrate state keys between schema versions.
          v1 → states had 3 components (density_tod_trend)
          v2 → same
          v3 → added mobility: density_tod_trend_mobility
        """
        parts = old_key.split('_')
        if from_schema < 3 and len(parts) == 3:
            # Assume stationary — safest default for migrated data
            return old_key + '_' + self.MOBILITY_STATIONARY
        return old_key

    # ─────────────────────────────────────────────────────────────────────
    # Time-of-day priors + hierarchical marginal priors
    # ─────────────────────────────────────────────────────────────────────

    def _apply_tod_prior(self):
        """
        Seed empty arms with weak synthetic observations so cold start
        is not random. Real data always dominates: n=1 prior vs n=40
        window means real observations win after 3-4 samples.
        """
        tod_priors = {
            'night': {
                'recon_time': 35, 'min_rssi': -80, 'max_interactions': 2,
                'hop_recon_time': 10, 'min_recon_time': 7,
                'throttle_d': 0.9, 'throttle_a': 0.4,
                'ap_ttl': 300, 'sta_ttl': 600,
                'max_misses_for_recon': 7, 'max_inactive_scale': 3,
                'recon_inactive_multiplier': 2,
            },
            'morning': {
                'recon_time': 25, 'min_rssi': -72, 'max_interactions': 3,
                'hop_recon_time': 8, 'min_recon_time': 5,
                'throttle_d': 0.7, 'throttle_a': 0.4,
                'ap_ttl': 180, 'sta_ttl': 300,
                'max_misses_for_recon': 5, 'max_inactive_scale': 2,
                'recon_inactive_multiplier': 2,
            },
            'afternoon': {
                'recon_time': 20, 'min_rssi': -72, 'max_interactions': 4,
                'hop_recon_time': 6, 'min_recon_time': 5,
                'throttle_d': 0.7, 'throttle_a': 0.4,
                'ap_ttl': 120, 'sta_ttl': 300,
                'max_misses_for_recon': 5, 'max_inactive_scale': 2,
                'recon_inactive_multiplier': 2,
            },
            'evening': {
                'recon_time': 20, 'min_rssi': -70, 'max_interactions': 5,
                'hop_recon_time': 6, 'min_recon_time': 3,
                'throttle_d': 0.5, 'throttle_a': 0.2,
                'ap_ttl': 120, 'sta_ttl': 300,
                'max_misses_for_recon': 5, 'max_inactive_scale': 2,
                'recon_inactive_multiplier': 2,
            },
        }
        # Mobility adjustments: mobile = shorter TTLs, shorter recon
        mobility_adjust = {
            self.MOBILITY_STATIONARY: {'ap_ttl': +60, 'sta_ttl': +120, 'recon_time': +5},
            self.MOBILITY_WALKING:    {},
            self.MOBILITY_MOBILE:     {'ap_ttl': -40, 'sta_ttl': -100, 'recon_time': -5},
        }
        PRIOR_R = 0.30

        for density in ('sparse', 'medium', 'dense'):
            for tod, vals in tod_priors.items():
                for trend in ('falling', 'stable', 'rising'):
                    for mobility in (self.MOBILITY_STATIONARY,
                                     self.MOBILITY_WALKING,
                                     self.MOBILITY_MOBILE):
                        state = f'{density}_{tod}_{trend}_{mobility}'
                        adj   = mobility_adjust.get(mobility, {})
                        for param, preferred in vals.items():
                            if param not in self.UCB_ARMS:
                                continue
                            pref = preferred + adj.get(param, 0)
                            self._ensure_state(param, state)
                            arms    = self.UCB_ARMS[param]
                            nearest = min(arms, key=lambda a: abs(a - pref))
                            entry   = self.ucb_table[param][state][nearest]
                            if entry['n'] == 0:
                                entry['n'] = 1
                                entry['rewards'].append(PRIOR_R)

    def _hierarchical_marginal(self, param, state):
        """
        Compute the marginal mean reward for a parameter, averaged
        across all states that share 2+ dimensions with the target
        state. This lets rare states benefit from common ones.

        Returns (mean, total_weight) or (None, 0) if insufficient data.
        """
        target_parts = state.split('_')
        if len(target_parts) != 4:
            return None, 0

        totals = defaultdict(lambda: [0.0, 0])   # arm -> [sum_reward, n]

        for other_state, arms in self.ucb_table.get(param, {}).items():
            other_parts = other_state.split('_')
            if len(other_parts) != 4 or other_state == state:
                continue
            # Count shared dimensions
            shared = sum(1 for a, b in zip(target_parts, other_parts) if a == b)
            if shared < 2:
                continue
            # Weight by similarity (3 shared = 1.0, 2 shared = 0.25)
            weight = 1.0 if shared == 3 else 0.25
            for arm, d in arms.items():
                if d['n'] > 0 and d['rewards']:
                    mean = sum(d['rewards']) / len(d['rewards'])
                    totals[arm][0] += mean * weight * d['n']
                    totals[arm][1] += weight * d['n']

        if not totals:
            return None, 0

        # Pick the arm with the highest weighted mean as the marginal best
        best_arm  = None
        best_mean = -1.0
        best_n    = 0
        for arm, (wsum, wn) in totals.items():
            if wn > 0:
                m = wsum / wn
                if m > best_mean:
                    best_mean = m
                    best_arm  = arm
                    best_n    = wn
        return best_arm, best_n

    # ─────────────────────────────────────────────────────────────────────
    # Context / state computation
    # ─────────────────────────────────────────────────────────────────────

    def _compute_mobility(self):
        """Infer mobility from GPS speed (if available) or AP turnover."""
        # GPS-based (preferred when available)
        if self._gps_last_fix is not None:
            speed = _sf(self._gps_last_fix.get('speed', 0))
            age   = time.monotonic() - self._gps_last_fix['ts_mono']
            if age <= self.cfg['gps_stale_seconds']:
                if speed >= self.cfg['mobility_mobile_threshold']:
                    return self.MOBILITY_MOBILE
                if speed >= self.cfg['mobility_walk_threshold']:
                    return self.MOBILITY_WALKING
                return self.MOBILITY_STATIONARY

        # Fallback: AP turnover heuristic
        # If many APs have appeared/disappeared recently, we're moving.
        if len(self._fp_history) >= 3:
            recent = list(self._fp_history)[-3:]
            # Count unique channels across the last 3 fingerprints
            all_chs = set()
            for fp in recent:
                all_chs.update(fp.get('top', []))
            if len(all_chs) >= 8:
                return self.MOBILITY_MOBILE
            if len(all_chs) >= 5:
                return self.MOBILITY_WALKING
        return self.MOBILITY_STATIONARY

    def _compute_state(self, aps_ema):
        """Map current observations to a discrete state key."""
        # AP density
        if aps_ema >= self.cfg['dense_aps']:
            density = 'dense'
        elif aps_ema <= self.cfg['sparse_aps']:
            density = 'sparse'
        else:
            density = 'medium'

        # Time of day (local time)
        h = time.localtime().tm_hour
        if h < 7:       tod = 'night'
        elif h < 12:    tod = 'morning'
        elif h < 18:    tod = 'afternoon'
        else:           tod = 'evening'

        # Reward trend (EMA delta)
        if self._reward_trend > 0.02:   trend = 'rising'
        elif self._reward_trend < -0.02: trend = 'falling'
        else:                            trend = 'stable'

        # Mobility (cached per-epoch)
        mobility = self._current_mobility

        return f'{density}_{tod}_{trend}_{mobility}'

    # ─────────────────────────────────────────────────────────────────────
    # UCB arm selection (Sliding-Window UCB1 with hierarchical fallback)
    # ─────────────────────────────────────────────────────────────────────

    def _ucb_select(self, param, state):
        """
        Pick best arm for (param, state).

        Algorithm:
          1. If any arm has zero observations (even after priors): try it.
          2. If the whole state is sparse (< 3 arms with data), use the
             hierarchical marginal from similar states.
          3. Otherwise: standard SW-UCB1 with annealed C.
        """
        # Cache: return cached choice if we computed this recently
        cache_key = (param, state)
        if (self._profile['ucb_cache_epochs'] > 0
                and cache_key in self._ucb_cache
                and self.epochs_seen < self._ucb_cache[cache_key][1]):
            return self._ucb_cache[cache_key][0]

        self._ensure_state(param, state)
        arms  = self.UCB_ARMS[param]
        table = self.ucb_table[param][state]

        # Try untried arms first (post-prior)
        untried = [a for a in arms if table[a]['n'] == 0]
        if untried:
            choice = random.choice(untried)
        else:
            # Shrinkage-aware UCB ALWAYS — it self-handles sparse cells
            # by pulling toward the parent group, so we no longer need a
            # separate hierarchical-marginal branch.
            choice = self._sw_ucb_pick_with_state(param, state, arms, table)

        # Cache
        if self._profile['ucb_cache_epochs'] > 0:
            self._ucb_cache[cache_key] = (
                choice, self.epochs_seen + self._profile['ucb_cache_epochs'])
        return choice

    def _arm_parent_mean(self, param, state, arm):
        """
        Empirical-Bayes prior for an arm: weighted mean of this arm's
        rewards across (a) states sharing 2+ context dims (preferred),
        falling back to (b) population mean across ALL states.

        Returns float prior, or None if there's no data anywhere.
        """
        target_parts = state.split('_')
        parent_sum = parent_n = 0.0
        pop_sum    = pop_n    = 0.0

        for other_state, arms in self.ucb_table.get(param, {}).items():
            d = arms.get(arm)
            if d is None or d['n'] == 0 or not d['rewards']:
                continue
            m = sum(d['rewards']) / len(d['rewards'])
            n = d['n']
            if other_state == state:
                continue   # we already have local in the caller
            pop_sum += m * n
            pop_n   += n
            op = other_state.split('_')
            if len(op) == 4 and len(target_parts) == 4:
                shared = sum(1 for a, b in zip(target_parts, op) if a == b)
                if shared >= 2:
                    w = 1.0 if shared == 3 else 0.4
                    parent_sum += m * w * n
                    parent_n   += w * n

        if parent_n > 0:
            return parent_sum / parent_n
        if pop_n > 0:
            return pop_sum / pop_n
        return None

    def _current_shrinkage_k(self):
        """
        Annealed shrinkage strength.

        With FIXED k=5 (v1.1) we observed in real telemetry that
        genuinely-better-but-undertested arms (e.g. min_rssi=-75 with
        n=13, local mean=0.40) had their effective mean pulled down to
        ~0.37 by the prior of 0.30 — slowing convergence.

        Solution: k starts heavy (k_max=5) for cold-start, then anneals
        linearly to k_min=1 as the table accumulates real samples. At
        the end of anneal, an arm with n=13 retains 93% of its local
        mean instead of 72%.

        If ucb_shrinkage_k is set explicitly in user config (legacy
        v1.1 behaviour), that fixed value is used and anneal is skipped.
        """
        fixed = self.cfg.get('ucb_shrinkage_k')
        if fixed is not None:
            try:
                return max(0.0, float(fixed))
            except (TypeError, ValueError):
                pass
        k_max = float(self.cfg.get('ucb_shrinkage_k_max', 5.0))
        k_min = float(self.cfg.get('ucb_shrinkage_k_min', 1.0))
        anneal = max(1, int(self.cfg.get('ucb_shrinkage_anneal_samples', 500)))
        # Count total real-sample volume across the entire UCB table.
        # This is independent of any single arm so cold-start cells
        # benefit from heavy k even after the table is mature.
        total = 0
        for states in self.ucb_table.values():
            for arms in states.values():
                for d in arms.values():
                    total += d.get('n', 0)
        frac = min(1.0, total / anneal)
        return k_max - (k_max - k_min) * frac

    def _sw_ucb_pick_with_state(self, param, state, arms, table):
        """
        Sliding-window UCB1 pick with annealed empirical-Bayes shrinkage.

        With 14 params × 108 states × ~6 arms = 9072 cells and a typical
        session of 100-300 epochs, vanilla UCB sees almost every cell as
        "the seeded prior". Shrinkage pulls each cell's mean toward the
        mean of similar states (sharing 2+ context dims) so neighbours
        contribute information. As the local sample count grows, the
        local mean reasserts itself via weight = n/(n+k); k itself
        anneals from heavy to light over the first ~500 real samples.
        """
        # Annealed exploration constant
        if self._exploration_boost > 0:
            C = self.cfg['exploration_boost_c']
        else:
            frac   = min(1.0, self.epochs_seen / self.cfg['ucb_c_anneal_epochs'])
            C_min  = self.cfg['ucb_c_floor']
            C_max  = self.cfg['ucb_c']
            C      = C_max - (C_max - C_min) * frac

        k = self._current_shrinkage_k()
        total_w = sum(len(table[a]['rewards']) for a in arms)
        best_score = -math.inf
        best_arm   = arms[0]
        for arm in arms:
            d      = table[arm]
            w_size = len(d['rewards'])
            local  = sum(d['rewards']) / w_size if w_size > 0 else 0.0
            prior  = self._arm_parent_mean(param, state, arm)
            if prior is None:
                # No information anywhere → cold-start neutral.
                eff = local if w_size > 0 else 0.3
            else:
                w_loc = w_size / (w_size + k)
                eff   = w_loc * local + (1.0 - w_loc) * prior
            expl   = C * math.sqrt(math.log(max(2, total_w)) / max(1, w_size))
            score  = eff + expl
            if score > best_score:
                best_score = score
                best_arm   = arm
        return best_arm

    # Back-compat shim: existing call sites pass (arms, table).
    def _sw_ucb_pick(self, arms, table):
        # Reconstruct (param, state) from caller's frame. Only used by old
        # callers that haven't been switched to the with_state form.
        # For correctness we just fall back to the un-shrunk path.
        if self._exploration_boost > 0:
            C = self.cfg['exploration_boost_c']
        else:
            frac   = min(1.0, self.epochs_seen / self.cfg['ucb_c_anneal_epochs'])
            C_min  = self.cfg['ucb_c_floor']
            C_max  = self.cfg['ucb_c']
            C      = C_max - (C_max - C_min) * frac
        total_w = sum(len(table[a]['rewards']) for a in arms)
        best_score = -math.inf
        best_arm   = arms[0]
        for arm in arms:
            d      = table[arm]
            w_size = len(d['rewards'])
            mean   = sum(d['rewards']) / w_size if w_size > 0 else 0.0
            expl   = C * math.sqrt(math.log(max(1, total_w)) / max(1, w_size))
            score  = mean + expl
            if score > best_score:
                best_score = score
                best_arm   = arm
        return best_arm

    def _ucb_update(self, param, state, arm, reward):
        """Record a reward observation for (param, state, arm)."""
        if param not in self._active_params:
            return
        self._ensure_state(param, state)
        tbl = self.ucb_table[param][state]
        if arm not in tbl:
            W = int(self._profile['ucb_window'])
            tbl[arm] = {'n': 0, 'rewards': deque(maxlen=W)}
        tbl[arm]['n'] += 1
        tbl[arm]['rewards'].append(float(reward))
        # Invalidate cache for this (param, state)
        self._ucb_cache.pop((param, state), None)

    # ─────────────────────────────────────────────────────────────────────
    # Custom handshake-focused reward (adaptive, percentile-based target)
    # ─────────────────────────────────────────────────────────────────────

    def _adaptive_hpm_target(self):
        """
        Adaptive reward target for unique handshakes per minute.

        We want a target that scales with what's achievable in the current
        environment, but doesn't move so fast it kills the reward signal.
        Strategy: use 90th percentile of recent hpm — only the very best
        recent epochs raise the bar. Floor at 0.5 unique per minute.
        """
        if len(self._recent_hpm) < 10:
            return 0.5  # default target: 0.5 unique/min ≈ 30/hour
        vals = sorted(self._recent_hpm)
        idx  = int(len(vals) * 0.90)
        p90  = vals[min(idx, len(vals) - 1)]
        # Never drop below a useful threshold; cap upper to prevent runaway
        return max(0.5, min(p90, 5.0))

    def _custom_reward(self, handshakes, hs_rate, missed_rate, native_reward,
                       duration_secs, lifetime_new_this_epoch,
                       active_ratio, inactive_ratio, hops_ratio,
                       new_aps_seen, attack_efficiency_proxy, interactions,
                       blind_ratio=0.0, bored_ratio=0.0, sad_ratio=0.0):
        """
        UNIQUE-handshake-maximising reward.

        The PRIMARY objective is lifetime-new BSSIDs per minute. Catching the
        same network 10 times = same reward as catching it once. Catching a
        brand-new network = full reward.

        Components (all normalised to [0,1]):
          - lifetime-new HS per minute         (0.60)  primary objective
          - new APs discovered this epoch      (0.10)  exploration value
          - unique-HS-per-attack efficiency    (0.08)  duplicates don't help
          - inverse missed rate                (0.06)  efficiency
          - pwnagotchi active ratio            (0.05)  signal we are working
          - channel hop diversity              (0.04)  coverage
          - native pwnagotchi reward           (0.03)  loose alignment
          - "underlying work" baseline         (0.04)  prevents 0-hs deadzones
                                                       from learning nothing
          - penalty: inactive ratio            (-0.05) penalty for stalls
          - penalty: blind ratio               (-0.07) radio sees nothing —
                                                      strong signal to
                                                      change scanning params
        Floor:
          - 0.01 if there were any interactions, so UCB still distinguishes
            'tried something but failed' from 'did nothing'.
        """
        dur_min        = max(0.01, duration_secs / 60.0)
        hs_per_min     = handshakes / dur_min
        new_per_min    = lifetime_new_this_epoch / dur_min
        # NB: stored in _recent_hpm but tracks UNIQUE-per-min, not total.
        self._recent_hpm.append(new_per_min)

        target = self._adaptive_hpm_target()

        # 1) PRIMARY: lifetime-new captures per minute against adaptive target.
        # Telemetry showed the previous log-scaling buried the signal:
        # ratio=1 (you HIT the target) only scored 0.20, indistinguishable
        # from the seeded UCB prior of 0.30 once weighted. UCB therefore
        # could not tell "good epoch" from "no data". Use a Hill-style
        # saturation r = ratio/(ratio+k) with k=1, which gives a sharp
        # gradient where it matters and gentle saturation above:
        #   ratio=0       → 0.00
        #   ratio=0.5     → 0.33
        #   ratio=1.0     → 0.50  (target hit — clearly above 0.30 prior)
        #   ratio=2.0     → 0.67
        #   ratio=4.0     → 0.80
        #   ratio=8.0     → 0.89
        if target > 0 and new_per_min > 0:
            ratio = new_per_min / target
            new_term = ratio / (ratio + 1.0)
        else:
            new_term = 0.0

        # 2) UNIQUE handshake efficiency per attack — duplicates don't count.
        # Catching the same AP 10× shouldn't beat catching 1 new AP once.
        eff_term = min(1.0, lifetime_new_this_epoch / max(1, interactions))

        # 3) Inverse missed rate
        miss_term = max(0.0, 1.0 - missed_rate)

        # 4) New APs discovered (exploration value, even without HS)
        new_aps_term = min(1.0, new_aps_seen / 10.0)

        # 5) Active ratio — pwnagotchi's own working signal
        active_term = min(1.0, active_ratio)

        # 6) Hop diversity
        hops_term = min(1.0, hops_ratio)

        # 7) Inactive penalty
        inactive_pen = min(1.0, inactive_ratio)

        # 7b) Blind penalty — radio "sees nothing" is a strong signal that
        # scan params (channels / hop_recon_time / recon_time) are wrong.
        # Original pwnagotchi AI used -0.30 here; we soften to -0.07 because
        # we already have an explicit blind-recovery state machine that
        # forces aggressive params, but keeping a small reward signal lets
        # UCB shy away from arms that lead to blindness.
        blind_pen = min(1.0, max(0.0, blind_ratio))

        # 7c) Mood penalties — pwnagotchi spends time bored/sad means it
        # was idle long enough to flip emotion. Original AI weighted these
        # at -0.20 / -0.10; we use smaller weights since blind_pen + the
        # active_term already cover most of the same signal.
        bored_pen = min(1.0, max(0.0, bored_ratio))
        sad_pen   = min(1.0, max(0.0, sad_ratio))

        # 8) Native reward
        native_term = min(1.0, max(0.0, _sf(native_reward)))

        # 9) "Underlying work" baseline — even when 0 HS, reward attempting
        # the right things (so UCB still learns in sparse environments)
        work_term = min(1.0, attack_efficiency_proxy)

        r = (
            0.60 * new_term         # this IS the goal
          + 0.10 * new_aps_term
          + 0.08 * eff_term
          + 0.06 * miss_term
          + 0.05 * active_term
          + 0.04 * hops_term
          - 0.05 * inactive_pen
          - 0.07 * blind_pen
          - 0.04 * sad_pen
          - 0.03 * bored_pen
          + 0.03 * native_term
          + 0.04 * work_term
        )
        # Activity floor: if we did anything at all this epoch, give UCB a
        # small but non-zero gradient. Prevents arms from looking equally
        # bad at 0.0 in long deadzones.
        if interactions > 0 and new_per_min == 0:
            r = max(r, 0.01)

        # Optional: stash component breakdown for debug logging. Avoids
        # recomputing in the on_epoch summary line.
        self._last_reward_breakdown = {
            'new':      0.60 * new_term,
            'new_aps':  0.10 * new_aps_term,
            'eff':      0.08 * eff_term,
            'miss':     0.06 * miss_term,
            'active':   0.05 * active_term,
            'hops':     0.04 * hops_term,
            'inact':   -0.05 * inactive_pen,
            'blind':   -0.07 * blind_pen,
            'sad':     -0.04 * sad_pen,
            'bored':   -0.03 * bored_pen,
            'native':   0.03 * native_term,
            'work':     0.04 * work_term,
        }
        return max(0.0, min(1.0, r))

    # ─────────────────────────────────────────────────────────────────────
    # EMA smoothing
    # ─────────────────────────────────────────────────────────────────────

    # Per-key sane clamps. ANY input outside these is treated as a bad
    # sample and the EMA is updated with the clamped value instead. Prevents
    # one rogue epoch (e.g. native pwnagotchi reward returning 1e16) from
    # poisoning the EMA forever — without that clamp we observed the
    # 'reward' EMA get stuck at -8.5e15 across 125 epochs.
    EMA_CLAMP = {
        'aps':            (0.0, 5000.0),
        'hs_rate':        (0.0, 1.0),
        'reward':         (-2.0, 2.0),
        'missed_rate':    (0.0, 1.0),
        'hs_per_min':     (0.0, 600.0),
        'active_ratio':   (0.0, 1.0),
        'inactive_ratio': (0.0, 1.0),
        'hops_per_epoch': (0.0, 200.0),
        'temperature':    (-40.0, 130.0),
        'cpu_load':       (0.0, 1.0),
        'speed':          (0.0, 200.0),
    }

    def _ema(self, key, value):
        # Reject non-finite (nan/inf) — they would propagate forever.
        v = _sf(value)
        if not math.isfinite(v):
            v = 0.0
        # Per-key clamp to reject pathological inputs.
        lo, hi = self.EMA_CLAMP.get(key, (-1e9, 1e9))
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        a    = float(self.cfg['ema_alpha'])
        prev = self.ema.get(key)
        # Defensive: if a stored EMA somehow got corrupted (NaN/inf, or far
        # outside the clamp range — e.g. legacy state from before this
        # safeguard), drop it and treat this sample as the first.
        if prev is not None:
            if not math.isfinite(_sf(prev)) or prev < lo - 1e-6 or prev > hi + 1e-6:
                prev = None
        new  = v if prev is None else (a * v + (1.0 - a) * prev)
        # Final defensive clamp on the output too.
        if new < lo:
            new = lo
        elif new > hi:
            new = hi
        self.ema[key] = new
        return new

    # ─────────────────────────────────────────────────────────────────────
    # Channel scoring & scheduling
    # ─────────────────────────────────────────────────────────────────────

    def _ch_score(self, ch):
        """
        Score a channel for priority selection.
        Combines persistent lifetime stats with fresh session signals
        AND a critical "uncaptured AP opportunity" boost.

        For UNIQUE handshakes, what matters is not just past success — it's
        whether there are CURRENTLY VISIBLE uncaptured APs on this channel.
        That signal dominates over historical data once we have it.
        """
        # FIX: take a single locked snapshot of all _ch_lt / _free_channels /
        # _gps_zones / _known_aps state we need; concurrent event-handlers
        # mutate these dicts/lists. After this we work on snapshots only.
        live_uncaptured           = 0
        live_uncaptured_w_clients = 0
        live_strong_signals       = 0
        with self._state_lock:
            lt        = dict(self._ch_lt[ch])
            free      = sum(1 for c in self._free_channels if c == ch)
            aps_snap  = list(self._known_aps.values())
            zone_ch_count = 0
            if self._current_zone is not None:
                zc = self._gps_zones.get(self._current_zone, {}).get(
                    'channels', {})
                zone_ch_count = zc.get(ch, 0)
            dead_count = self._dead_lt.get(ch, 0)
        for ap in aps_snap:
            if _si(ap.get('channel', 0)) != ch:
                continue
            if not ap.get('AT_visible', False):
                continue
            if ap.get('AT_already_captured', False):
                continue
            if ap.get('AT_pmf_detected', False):
                continue
            if ap.get('AT_cooldown_until', 0) > self.epochs_seen:
                continue
            live_uncaptured += 1
            if ap.get('AT_clients', 0) > 0:
                live_uncaptured_w_clients += 1
            if _sf(ap.get('rssi', -100)) > -70:
                live_strong_signals += 1

        score = (
            # Historical productivity (lifetime stats)
            lt['hs']         * 4.0
          + lt['passive_hs'] * 5.0       # passive captures are FREE — boost
          + lt['cracked']    * 1.0       # cracked = confirmed weak password area
          - lt['wasted']     * 0.7
          + lt['free_seen']  * 0.3

            # Live opportunity (current uncaptured APs visible) — DOMINANT
          + live_uncaptured            * 6.0
          + live_uncaptured_w_clients  * 4.0   # clients = deauth opportunity
          + live_strong_signals        * 3.0   # strong RSSI = high success rate

            # Channel positioning
          + free             * 2.0
          + (2.0 if ch in self.NON_OVERLAPPING else 0.0)
          + 0.01
        )

        # Zone-specific channel bonus: if this channel has been
        # productive in the current GPS zone specifically
        if zone_ch_count:
            score += zone_ch_count * 1.5

        # Channel-efficiency multiplier — telemetry showed the plugin
        # was over-visiting low-yield channels (ch1: 5% HS/attack rate
        # over 332 assocs vs ch8: 73% HS/attack rate over 41 assocs).
        # Once a channel has enough samples to judge it, scale by
        # success rate so the better channel keeps winning the auction.
        # Floor 0.5× / cap 1.5× — never zero out the channel entirely.
        attempts_lt = lt['assocs'] + lt['deauths']
        if attempts_lt >= 30:
            eff = lt['hs'] / max(1, attempts_lt)
            # eff=0.05 → mul=0.65, eff=0.20 → mul=1.10, eff=0.35 → mul=1.45
            mul = max(0.5, min(1.5, 0.5 + eff * 3.0))
            score *= mul

        score *= max(0.05, 1.0 - float(self.cfg['dead_ch_lifetime_weight'])
                                 * dead_count)
        return max(0.0, score)

    def _pick_weighted(self, pool, n):
        """Weighted random draw without replacement, weighted by score."""
        if not pool or n <= 0:
            return []
        candidates = [(c, self._ch_score(c)) for c in pool]
        total      = sum(s for _, s in candidates)
        picks      = []
        while len(picks) < n and candidates and total > 1e-9:
            r   = random.random() * total
            acc = 0.0
            for i, (c, s) in enumerate(candidates):
                acc += s
                if acc >= r:
                    picks.append(c)
                    total -= s
                    candidates.pop(i)
                    break
        return picks

    def _schedule_channels(self, agent):
        """Build the next scan channel list (dedup, score-sorted)."""
        try:
            n_extra = int(self._profile['extra_channels'])

            if not self._unscanned_channels:
                if 'restrict_channels' in self.cfg:
                    pool = list(self.cfg['restrict_channels'])
                elif hasattr(agent, '_allowed_channels') and agent._allowed_channels:
                    pool = list(agent._allowed_channels)
                elif hasattr(agent, '_supported_channels') and agent._supported_channels:
                    pool = list(agent._supported_channels)
                else:
                    try:
                        pool = pwnagotchi.utils.iface_channels(
                            agent._config['main']['iface'])
                    except Exception:
                        pool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                self._unscanned_channels = pool

            cdl       = int(self.cfg['dead_channel_cooldown'])
            available = [c for c in self._unscanned_channels
                         if self._dead_session.get(c, 0) < cdl]
            if not available:
                self._dead_session.clear()
                available = list(self._unscanned_channels)

            pw     = float(self.cfg['priority_channel_weight'])
            n_prio = max(1, int(round(n_extra * pw))) if n_extra > 0 else 0
            n_expl = max(0, n_extra - n_prio)

            prio_pool  = [c for c in available if self._ch_score(c) > 0.05]
            prio_picks = self._pick_weighted(prio_pool, n_prio)

            shortfall  = n_prio - len(prio_picks)
            expl_pool  = [c for c in available if c not in prio_picks]
            n_expl    += shortfall
            expl_picks = (random.sample(expl_pool, min(n_expl, len(expl_pool)))
                          if expl_pool else [])

            # Build score-sorted, deduplicated channel list
            all_candidates = set(self._active_channels) | set(prio_picks) | set(expl_picks)
            next_chs = sorted(
                all_candidates,
                key=lambda c: -self._ch_score(c),
            )

            # CRITICAL: cap total channels to prevent recon stalls in dense
            # environments. Each channel gets ~hop_recon_time seconds. If we
            # have 15 channels in the list with hop_recon_time=10, that's
            # 2.5 minutes per recon cycle = lots of missed handshakes.
            # Cap = base_active(max 6) + extra_channels.
            personality = agent._config.get('personality', {})
            hrt = _sf(personality.get('hop_recon_time', 8))
            # Calculate sane max so one full recon cycle is bounded
            # to ~60-90 seconds even at high hrt
            max_active_in_list = min(6, len(self._active_channels))
            max_total = max_active_in_list + n_extra

            # Identify channels with strong, attackable, uncaptured APs.
            # These MUST stay in the list even if score-cap would drop them —
            # losing such a channel = guaranteed missed unique handshake.
            must_keep = set()
            with self._state_lock:
                for ap in list(self._known_aps.values()):
                    if not ap.get('AT_visible'):
                        continue
                    if ap.get('AT_already_captured'):
                        continue
                    if ap.get('AT_pmf_detected'):
                        continue
                    if ap.get('AT_cooldown_until', 0) > self.epochs_seen:
                        continue
                    if _sf(ap.get('rssi', -100)) <= -75:
                        continue
                    ch = _si(ap.get('channel', 0))
                    if ch:
                        must_keep.add(ch)

            if len(next_chs) > max_total:
                # Keep top max_total by score, then add must_keep channels
                # that didn't make the cap (force-included)
                top_keep      = next_chs[:max_total]
                forced_extras = [c for c in must_keep if c not in top_keep]
                final         = top_keep + forced_extras
                dropped       = [c for c in next_chs if c not in final]
                next_chs      = final
                # Push dropped channels back to unscanned for next cycle
                for ch in dropped:
                    if ch not in self._unscanned_channels and ch not in next_chs:
                        self._unscanned_channels.append(ch)

            # Remove used extras from unscanned pool
            for ch in prio_picks + expl_picks:
                if ch in self._unscanned_channels and ch in next_chs:
                    self._unscanned_channels.remove(ch)

            agent._config['personality']['channels'] = next_chs
            # Note: channel set is applied by pwnagotchi's recon loop
            # via `wifi.recon.channel` — no manual sync needed.
        except Exception as e:
            logging.exception(f'[envtune] _schedule_channels: {e}')

    # ─────────────────────────────────────────────────────────────────────
    # AP tracking (thread-safe via _state_lock on public entry points)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _norm(name):
        if not name:
            return 'EMPTY'
        return ''.join(c for c in str(name).lower() if c.isalnum())

    def _ap_id(self, ap):
        """
        Unique AP identifier. For hidden SSIDs we use only the MAC so
        each hidden AP is tracked separately (previous versions merged
        them under a single 'HIDDEN' id, losing data).
        """
        hostname = ap.get('hostname', '')
        mac      = ap.get('mac', '')
        if not hostname or hostname == '<hidden>' or hostname == '':
            return 'hidden-' + self._mac_norm(mac)
        return self._norm(hostname) + '-' + self._mac_norm(mac)

    @staticmethod
    def _mac_norm(mac):
        return str(mac).lower().replace(':', '').replace('-', '').replace(' ', '')

    def _is_whitelisted(self, ap):
        """Check if an AP is in the user's whitelist (SSID or MAC)."""
        mac_n = self._mac_norm(ap.get('mac', ''))
        if mac_n and mac_n in self._whitelist_macs:
            return True
        ssid = ap.get('hostname', '')
        if ssid and ssid in self._whitelist_ssids:
            return True
        # Partial MAC prefix match (pwnagotchi supports this)
        for prefix in self._whitelist_macs:
            if len(prefix) < 12 and mac_n.startswith(prefix):
                return True
        return False

    def _mark_ap_seen(self, ap, context=None):
        try:
            # Skip whitelisted APs entirely — pwnagotchi won't attack them
            # so tracking them would distort channel scores
            if self._is_whitelisted(ap):
                return

            # Enforce AP dict size cap (evict least-recently-seen)
            cap = int(self._profile['ap_track_max'])
            if len(self._known_aps) >= cap:
                self._evict_oldest_ap()

            apID  = self._ap_id(ap)
            ch    = _si(ap.get('channel', 0))
            tag   = 'AT_' + context if context else 'AT_seen'
            mac_n = self._mac_norm(ap.get('mac', ''))

            with self._state_lock:
                if apID not in self._known_aps:
                    entry = dict(ap)
                    entry.update({
                        'AT_seen':             1,
                        'AT_visible':          True,
                        'AT_attacks':          0,
                        'AT_handshake':        0,
                        'AT_clients':          0,
                        'AT_client_epoch':     -99,
                        'AT_lastattack_ep':    -99,
                        'AT_missed':           0,
                        'AT_rssi_hist':        deque([_sf(ap.get('rssi', -80))], maxlen=4),
                        'AT_pmf_detected':     False,
                        'AT_pmkid_success':    False,
                        'AT_cooldown_until':   0,
                        'AT_already_captured': mac_n in self._captured_bssids,
                        'AT_cracked':          mac_n in self._cracked_bssids,
                        'AT_efficiency':       0.0,
                        'AT_lastseen':         time.monotonic(),
                        # First-seen epoch — used by _ap_priority_score to
                        # surface brand-new BSSIDs aggressively. A BSSID
                        # appearing for the first time this session AND
                        # not in our captured set is the highest-EV moment
                        # to attack: clients may still be in connect/roam
                        # phase leaking EAPOL.
                        'AT_first_seen_ep':    self.epochs_seen,
                        tag:                   1,
                    })
                    self._known_aps[apID] = entry
                    self._inc_ch('Unique APs', ch)
                    self._inc_ch('Current APs', ch)
                else:
                    entry = self._known_aps[apID]
                    for k in ('rssi', 'hostname', 'channel', 'encryption', 'clients'):
                        if k in ap:
                            entry[k] = ap[k]
                    if not entry.get('AT_visible', True):
                        entry['AT_visible'] = True
                        entry['AT_seen']    = entry.get('AT_seen', 0) + 1
                        self._inc_ch('Current APs', ch)
                    entry[tag] = entry.get(tag, 0) + 1
                    entry['AT_rssi_hist'].append(_sf(ap.get('rssi', -80)))
                    if mac_n in self._captured_bssids:
                        entry['AT_already_captured'] = True
                    if mac_n in self._cracked_bssids:
                        entry['AT_cracked'] = True

                self._known_aps[apID]['AT_lastseen'] = time.monotonic()

        except Exception as e:
            logging.debug(f'[envtune] _mark_ap_seen: {e}')

    def _evict_oldest_ap(self):
        """Remove the lowest-value AP entry to stay under cap.

        Eviction priority (lowest value first):
          1. Already-captured APs (we won't attack them again — only
             retain for the skip-list, which lives in _bcap_skip_macs).
          2. Cracked APs (we have the password — no further value).
          3. APs with zero attack efficiency and many attacks (PMF / hidden /
             unreachable — wasting our attack budget if kept).
          4. Otherwise: oldest AT_lastseen (LRU fallback).

        Never evicts a fresh AP we haven't tried yet, even if it's "old".
        """
        if not self._known_aps:
            return

        def _score(k):
            ap          = self._known_aps[k]
            captured    = ap.get('AT_already_captured', False)
            cracked     = ap.get('AT_cracked', False)
            attacks     = ap.get('AT_attacks', 0)
            handshakes  = ap.get('AT_handshake', 0)
            efficiency  = ap.get('AT_efficiency', 0.0) or 0.0
            lastseen    = ap.get('AT_lastseen', 0)
            # Lower tuple = first to evict.
            # Tier 0: cracked + captured                         (cheapest)
            # Tier 1: captured but not cracked
            # Tier 2: many attacks, zero handshakes (dead target)
            # Tier 3: attacked but low-efficiency
            # Tier 4: never attacked (precious — keep)
            if captured and cracked:
                tier = 0
            elif captured:
                tier = 1
            elif attacks >= 5 and handshakes == 0:
                tier = 2
            elif attacks > 0 and efficiency < 0.05:
                tier = 3
            else:
                tier = 4
            return (tier, lastseen)
        with self._state_lock:
            victim = min(self._known_aps, key=_score)
            self._known_aps.pop(victim, None)

    def _rssi_trend(self, apID):
        """Positive = approaching (RSSI improving)."""
        ap = self._known_aps.get(apID)
        if not ap:
            return 0.0
        hist = list(ap.get('AT_rssi_hist', []))
        if len(hist) < 2:
            return 0.0
        return _sf(hist[-1]) - _sf(hist[0])

    def _ap_priority_score(self, apID):
        """Attack priority — higher = attack this first. 0 = skip."""
        ap = self._known_aps.get(apID)
        if ap is None:
            return 0.0
        if ap.get('AT_already_captured', False):
            return 0.02     # low but non-zero (new opportunities possible)
        if ap.get('AT_cooldown_until', 0) > self.epochs_seen:
            return 0.0

        score        = 1.0
        attacks      = ap.get('AT_attacks', 0)
        clients      = ap.get('AT_clients', 0)
        recency      = self.epochs_seen - ap.get('AT_client_epoch', -99)
        client_fresh = recency <= int(self.cfg['client_recency_epochs'])
        if clients > 0 and client_fresh:
            score += 4.0 * min(clients, 5)
            # FIX: a freshly-seen client on a still-uncaptured AP is the
            # single highest-value moment — clients leak EAPOL on connect/
            # roam and we want to be hammering deauth right then. Stack
            # an extra bonus when the client signal is *very* fresh.
            if recency <= 1:
                score += 2.0 * min(clients, 3)

        # Untried bonus — UCB-style optimism, encourage exploration of
        # APs we have never attacked. Decays as attacks accumulate.
        if attacks == 0:
            score += 1.0
        elif attacks <= 2:
            score += 0.5

        # Fresh-session bonus — a BSSID that appeared for the first time
        # this session is high-EV for unique captures. Decays linearly
        # over the next 8 epochs so the boost is real but doesn't
        # dominate forever. Skipped if we've already captured it.
        first_seen = ap.get('AT_first_seen_ep', -99)
        if first_seen >= 0:
            age = self.epochs_seen - first_seen
            if 0 <= age < 8:
                score += 1.5 * (1.0 - age / 8.0)

        score += ap.get('AT_efficiency', 0.0) * 3.0
        rssi   = _sf(ap.get('rssi', -85))
        score += max(0.0, (rssi + 65.0) / 20.0)
        score += self._rssi_trend(apID) * 0.4

        if ap.get('AT_pmf_detected', False) and not ap.get('AT_pmkid_success', False):
            score *= 0.2

        # Cracked networks are lower priority (we have the password)
        if ap.get('AT_cracked', False):
            score *= 0.3

        # Many attacks, zero handshakes → likely PMF/hidden — drop priority
        # before sinking more radio time. Don't go to zero (PMKID may still
        # work after a roam) but mute aggressively.
        if attacks >= 8 and ap.get('AT_handshake', 0) == 0:
            score *= 0.25

        return max(0.0, score)

    def _inc_ch(self, stat, ch, count=1):
        if stat not in self._chistos:
            self._chistos[stat] = {-1: 0}
        self._chistos[stat][ch] = self._chistos[stat].get(ch, 0) + count
        self._chistos[stat][-1] = self._chistos[stat].get(-1, 0) + count
        aa       = self._chistos['_all_actions']
        aa[ch]   = aa.get(ch, 0) + abs(count)
        aa[-1]   = aa.get(-1, 0) + abs(count)

    # ─────────────────────────────────────────────────────────────────────
    # Nexmon crash detection (uses pre-epoch-update EMA)
    # ─────────────────────────────────────────────────────────────────────

    def _check_nexmon_crash(self, aps, interactions):
        """Compare against prev_aps_ema (stored across epochs)."""
        prev = self._prev_aps_ema
        if prev is not None and prev > 5 and aps == 0 and interactions == 0:
            self._crash_suspect += 1
        else:
            self._crash_suspect = 0
        return self._crash_suspect >= 2

    # ─────────────────────────────────────────────────────────────────────
    # Location change detection (works with or without GPS)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_location_fp(self, access_points):
        if not access_points:
            return None
        ctr   = defaultdict(int)
        rssis = []
        for ap in access_points:
            ch = _si(ap.get('channel', 0))
            if ch > 0:
                ctr[ch] += 1
            rssis.append(_sf(ap.get('rssi', -80)))
        top = sorted(ctr.items(), key=lambda x: -x[1])[:5]
        return {
            'top':      [c for c, _ in top],
            'avg_rssi': sum(rssis) / len(rssis) if rssis else -80.0,
            'count':    len(access_points),
        }

    def _check_location_change(self, fp):
        if not fp:
            return False
        self._fp_history.append(fp)
        if self._loc_fp_stored is None:
            self._loc_fp_stored = fp
            return False
        old    = self._loc_fp_stored
        union  = set(fp['top']) | set(old['top'])
        jac    = (len(set(fp['top']) & set(old['top'])) / max(1, len(union))
                  if union else 1.0)
        rdiff  = abs(fp['avg_rssi'] - old['avg_rssi'])
        cratio = abs(fp['count'] - old['count']) / max(1, old['count'])
        moved  = jac < 0.30 or rdiff > 15.0 or cratio > 0.70
        self._loc_fp_stored = fp
        # Debounce: prevents constant retriggering while walking/driving
        # (zone hops every ~30s would otherwise reset the boost forever
        # and UCB would never reach exploit phase).
        if moved and self.epochs_seen - self._last_loc_change_ep < 5:
            return False
        if moved:
            self._last_loc_change_ep = self.epochs_seen
        return moved

    # ─────────────────────────────────────────────────────────────────────
    # GPS integration (TheyLive / stock gps / none)
    # ─────────────────────────────────────────────────────────────────────

    def _detect_gps_source(self, agent):
        """
        Determine how to read GPS. Preference order:
          1. agent.session()['gps'] (works for TheyLive + stock gps)
          2. TheyLive NDJSON track file (last line)
          3. None / disabled
        """
        if not self.cfg.get('enable_gps', True):
            return None
        try:
            session = agent.session() or {}
            if 'gps' in session and session['gps']:
                gps = session['gps']
                # Both TheyLive and stock gps expose lat/lon
                lat = _sf(gps.get('Latitude', gps.get('lat', 0)))
                lon = _sf(gps.get('Longitude', gps.get('lon', 0)))
                if lat != 0.0 or lon != 0.0:
                    return 'session'
        except Exception:
            pass
        if os.path.exists(self.GPS_TRACK):
            return 'theylive_ndjson'
        return None

    def _read_gps(self, agent):
        """
        Return current GPS fix dict or None.
        Dict format: {'lat', 'lon', 'alt', 'speed', 'ts_mono', 'raw'}
        """
        if self._gps_source is None:
            return None

        try:
            if self._gps_source == 'session':
                session = agent.session() or {}
                gps = session.get('gps') or {}
                if not gps:
                    return None
                lat   = _sf(gps.get('Latitude', gps.get('lat', 0)))
                lon   = _sf(gps.get('Longitude', gps.get('lon', 0)))
                if lat == 0.0 and lon == 0.0:
                    return None  # no lock
                alt   = _sf(gps.get('Altitude', gps.get('alt', 0)))
                speed = _sf(gps.get('Speed', gps.get('speed', 0)))
                # TheyLive also exposes 'track', 'hdop' — preserve raw
                return {
                    'lat': lat, 'lon': lon, 'alt': alt, 'speed': speed,
                    'ts_mono': time.monotonic(),
                    'raw': gps,
                }

            if self._gps_source == 'theylive_ndjson':
                # Read last line of NDJSON track file
                try:
                    with open(self.GPS_TRACK, 'rb') as f:
                        f.seek(0, 2)
                        size = f.tell()
                        if size == 0:
                            return None
                        # Read last ~4KB and find last \n
                        read_n = min(4096, size)
                        f.seek(size - read_n)
                        tail = f.read().decode('utf-8', errors='ignore')
                    last_line = tail.strip().split('\n')[-1] if tail.strip() else None
                    if not last_line:
                        return None
                    data = json.loads(last_line)
                    lat = _sf(data.get('lat', 0))
                    lon = _sf(data.get('lon', 0))
                    if lat == 0.0 and lon == 0.0:
                        return None
                    return {
                        'lat': lat, 'lon': lon,
                        'alt': _sf(data.get('alt', 0)),
                        'speed': _sf(data.get('speed', 0)),
                        'ts_mono': time.monotonic(),
                        'raw': data,
                    }
                except (FileNotFoundError, json.JSONDecodeError, ValueError):
                    return None

        except Exception as e:
            logging.debug(f'[envtune] _read_gps: {e}')

        return None

    def _zone_key(self, lat, lon):
        """
        Convert (lat, lon) into a string zone ID at configured resolution.
        Uses a simple grid: each cell ≈ resolution_m on a side.
        """
        res_m = float(self._profile['zone_resolution_m'])
        # 1 degree latitude  ≈ 111_000 m
        # 1 degree longitude ≈ 111_000 * cos(lat) m
        lat_cell = res_m / 111000.0
        lon_cell = res_m / max(1.0, 111000.0 * math.cos(math.radians(lat)))
        lat_idx  = int(math.floor(lat / lat_cell))
        lon_idx  = int(math.floor(lon / lon_cell))
        return f'{lat_idx}:{lon_idx}'

    GPS_ZONE_CAP = 500   # LRU cap to keep state file bounded
    # Tier-based zone eviction: zones with many attacks and zero
    # handshakes are confirmed-dead and should be dropped before
    # never-touched zones (which might produce handshakes later).
    # Telemetry showed 12/17 zones in real use with 0 HS, several with
    # 5+ visits and 20+ attacks, accumulating indefinitely.
    ZONE_DEAD_ATTACKS = 50

    def _update_gps_zone(self):
        """Update self._current_zone from current GPS fix."""
        fix = self._gps_last_fix
        if not fix:
            self._current_zone = None
            return
        age = time.monotonic() - fix['ts_mono']
        if age > self.cfg['gps_stale_seconds']:
            self._current_zone = None
            return
        zone = self._zone_key(fix['lat'], fix['lon'])
        self._current_zone = zone
        # FIX: _gps_zones is mutated here AND in on_handshake under lock,
        # AND read by _ch_score / _build_state_snapshot. Lock both writers.
        with self._state_lock:
            self._gps_zones[zone]['visits'] += 1
            self._gps_zones[zone]['last_seen'] = time.time()
            # LRU cap with tier-based eviction. Never evict the current
            # zone or any zone that has produced handshakes.
            #   tier 0: confirmed-dead (>=50 attacks, 0 HS) — evict FIRST
            #   tier 1: low-attack 0-HS zones (visited but not exhausted)
            #
            # Within each tier, oldest last_seen goes first (LRU).
            if len(self._gps_zones) > self.GPS_ZONE_CAP:
                def _zone_evict_key(item):
                    zk, zd = item
                    attacks = zd.get('attacks', 0) or 0
                    tier = 0 if attacks >= self.ZONE_DEAD_ATTACKS else 1
                    return (tier, zd.get('last_seen', 0.0) or 0.0)
                victims = [
                    (zk, zd) for zk, zd in self._gps_zones.items()
                    if zk != zone and zd.get('hs', 0) == 0
                ]
                victims.sort(key=_zone_evict_key)
                evict_n = len(self._gps_zones) - self.GPS_ZONE_CAP
                for zk, _zd in victims[:evict_n]:
                    self._gps_zones.pop(zk, None)

    # ─────────────────────────────────────────────────────────────────────
    # Parameter coupling — extensive sanity rules
    # ─────────────────────────────────────────────────────────────────────

    def _sanity_check(self, params):
        """
        Fix known bad inter-parameter combinations. UCB treats params as
        independent but some combinations are always wrong (e.g. very
        high recon_time with very low hop_recon_time).
        """
        p = dict(params)

        # 1) recon_time must be >= 2 × hop_recon_time
        rt  = _sf(p.get('recon_time', 25))
        hrt = _sf(p.get('hop_recon_time', 8))
        if rt < hrt * 2:
            p['recon_time'] = int(min(self.BOUNDS['recon_time'][1], hrt * 2))

        # 2) min_recon_time <= hop_recon_time
        mrt = _sf(p.get('min_recon_time', 5))
        if mrt > hrt:
            p['min_recon_time'] = int(hrt)

        # 3) sta_ttl >= ap_ttl (clients don't expire before their AP)
        if _sf(p.get('sta_ttl', 300)) < _sf(p.get('ap_ttl', 120)):
            p['sta_ttl'] = int(_sf(p.get('ap_ttl', 120)))

        # 4) Tight min_rssi + high max_interactions is wasteful
        if _sf(p.get('min_rssi', -75)) >= -67 and _si(p.get('max_interactions', 3)) > 5:
            p['max_interactions'] = 5

        # 5) Low throttle_d + high max_interactions risks nexmon crash
        if (_sf(p.get('throttle_d', 0.9)) < 0.5
                and _si(p.get('max_interactions', 3)) > 4):
            p['max_interactions'] = 4

        # 6) In mobile mode, long TTLs waste memory on out-of-range APs
        if self._current_mobility == self.MOBILITY_MOBILE:
            if _sf(p.get('ap_ttl', 120)) > 180:
                p['ap_ttl'] = 180
            if _sf(p.get('sta_ttl', 300)) > 300:
                p['sta_ttl'] = 300

        # 7) In stationary mode, short TTLs lose context unnecessarily
        if self._current_mobility == self.MOBILITY_STATIONARY:
            if _sf(p.get('ap_ttl', 120)) < 120:
                p['ap_ttl'] = 120

        # 8) max_misses_for_recon must allow for weak environments
        if self.ema.get('aps') is not None and _sf(self.ema.get('aps')) < 3:
            # Sparse environment: don't over-trigger recon on misses
            if _si(p.get('max_misses_for_recon', 5)) < 7:
                p['max_misses_for_recon'] = 7

        # 9) max_inactive_scale with very high recon_inactive_multiplier
        #    creates stalls (multiplier^scale × recon_time seconds)
        scale  = _si(p.get('max_inactive_scale', 2))
        mult   = _si(p.get('recon_inactive_multiplier', 2))
        if scale * mult > 8:
            p['max_inactive_scale'] = min(scale, 3)
            p['recon_inactive_multiplier'] = min(mult, 2)

        # 10) Low throttle_a under thermal pressure is dangerous
        if self._thermal_throttle and _sf(p.get('throttle_a', 0.4)) < 0.4:
            p['throttle_a'] = 0.4

        # 11) Very sparse environments: allow deeper min_rssi
        if self.ema.get('aps') is not None and _sf(self.ema.get('aps')) < 4:
            # Don't let the tuner tighten min_rssi when we barely see anything
            if _sf(p.get('min_rssi', -75)) > -75:
                p['min_rssi'] = -80

        # 12) Very dense environments: allow more aggressive filtering
        if self.ema.get('aps') is not None and _sf(self.ema.get('aps')) > 40:
            # Too many APs — focus on strong signals
            if _sf(p.get('min_rssi', -75)) < -78:
                p['min_rssi'] = -75

        # 13) During location change (exploration boost active) ease up
        if self._exploration_boost > 0:
            if _si(p.get('max_interactions', 3)) > 4:
                p['max_interactions'] = 4

        # 14) Low battery: reduce aggression
        if self._battery_level is not None:
            if self._battery_level < self.cfg['battery_critical_threshold']:
                p['max_interactions'] = min(_si(p.get('max_interactions', 3)), 2)
                if 'throttle_d' in self._active_params:
                    p['throttle_d'] = max(_sf(p.get('throttle_d', 0.9)), 0.9)

        # 15) Sad/bored mood: longer TTLs to catch slow activity
        if self._mood in ('sad', 'bored'):
            if _sf(p.get('sta_ttl', 300)) < 400:
                p['sta_ttl'] = 400

        # 16) 5GHz-aware recon_time. 5GHz handshakes complete faster (wider
        # channels, stronger short-range signals) — when 5GHz APs make up
        # a meaningful share of what we see, long recon_time wastes a cycle
        # we could spend hopping. Snapshot _ch_lt under lock to avoid races.
        try:
            with self._state_lock:
                hs_5    = sum(d['hs'] for ch, d in self._ch_lt.items() if ch >= 36)
                hs_24   = sum(d['hs'] for ch, d in self._ch_lt.items() if ch < 36)
                aps_5   = sum(1 for ch in self._ch_lt if ch >= 36
                              and self._ch_lt[ch].get('visits', 0) > 0)
                aps_24  = sum(1 for ch in self._ch_lt if ch < 36
                              and self._ch_lt[ch].get('visits', 0) > 0)
            tot_aps = aps_5 + aps_24
            if tot_aps >= 5 and (aps_5 / tot_aps) > 0.30:
                # Drop recon_time by 5s (clamped to bounds)
                target_rt = max(self.BOUNDS['recon_time'][0],
                                _sf(p.get('recon_time', 25)) - 5)
                if _sf(p.get('recon_time', 25)) > target_rt:
                    p['recon_time'] = int(target_rt)
        except Exception:
            pass

        return p

    # ─────────────────────────────────────────────────────────────────────
    # Stagnation check using rolling median
    # ─────────────────────────────────────────────────────────────────────

    def _check_stagnation(self, custom_rwd):
        self._reward_history.append(custom_rwd)
        if len(self._reward_history) < 10:
            return
        # Rolling median — outliers don't lock us into permanent stagnation
        sorted_r = sorted(self._reward_history)
        median   = sorted_r[len(sorted_r) // 2]
        if custom_rwd < median - 0.08:
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0
        if (self._stagnation_count >= int(self.cfg['stagnation_epochs'])
                and self._exploration_boost <= 0):
            self._exploration_boost = int(self.cfg['exploration_boost_epochs'])
            self._stagnation_count  = 0
            # Also clear cache so UCB recomputes
            self._ucb_cache.clear()
            # FIX: queued decisions were made under the stagnant policy;
            # crediting them with rewards from the boost period would
            # reinforce the very arms we want to escape. Drop the queue.
            self._decision_buffer.clear()
            logging.info(f'[envtune] Stagnation → '
                         f'{self._exploration_boost}-ep exploration boost')

    # ─────────────────────────────────────────────────────────────────────
    # Thermal safety
    # ─────────────────────────────────────────────────────────────────────

    def _apply_thermal_throttle(self, agent, temp):
        """Back off radio work when CPU temperature climbs."""
        p = agent._config['personality']
        if temp >= self.cfg['temp_critical']:
            self._thermal_throttle = True
            if 'throttle_d' in self._active_params:
                p['throttle_d'] = min(self.BOUNDS['throttle_d'][1],
                                      _sf(p.get('throttle_d', 0.9)) + 0.3)
            if 'throttle_a' in self._active_params:
                p['throttle_a'] = min(self.BOUNDS['throttle_a'][1],
                                      _sf(p.get('throttle_a', 0.4)) + 0.2)
            p['max_interactions'] = max(2, _si(p.get('max_interactions', 3)) - 1)
            logging.warning(f'[envtune] THERMAL CRITICAL {temp:.1f}°C — throttling')
        elif temp >= self.cfg['temp_warn']:
            self._thermal_throttle = True
            if 'throttle_d' in self._active_params:
                p['throttle_d'] = max(_sf(p.get('throttle_d', 0.9)), 0.9)
            logging.info(f'[envtune] Thermal warning {temp:.1f}°C')
        else:
            self._thermal_throttle = False

    # ─────────────────────────────────────────────────────────────────────
    # Battery integration (pisugar via UI element, if present)
    # ─────────────────────────────────────────────────────────────────────

    def _read_battery(self):
        """Read battery % from pisugar UI element if available."""
        if self._ui is None:
            return None
        try:
            bat = self._ui.get('bat')
            if not bat:
                return None
            # pisugar format: "50%" or similar
            s = str(bat).strip().rstrip('%').strip()
            if s.replace('.', '').isdigit():
                return _sf(s)
        except Exception:
            pass
        return None

    # ─────────────────────────────────────────────────────────────────────
    # wpa-sec cracked potfile feedback
    # ─────────────────────────────────────────────────────────────────────

    def _scan_cracked_potfile(self):
        """
        Read wpa-sec potfile if present. Format: BSSID:CLIENT:SSID:PASSWORD
        Returns a set of cracked BSSIDs (normalised).
        """
        cracked = set()
        if not self.cfg.get('enable_wpasec_feedback', True):
            return cracked
        try:
            if not os.path.exists(self.WPASEC_POT):
                return cracked
            with open(self.WPASEC_POT, 'r', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                    parts = line.split(':')
                    if parts:
                        mac = self._mac_norm(parts[0])
                        if len(mac) == 12 and all(c in '0123456789abcdef' for c in mac):
                            cracked.add(mac)
        except Exception as e:
            logging.debug(f'[envtune] potfile scan: {e}')
        return cracked

    # ─────────────────────────────────────────────────────────────────────
    # Whitelist loading from pwnagotchi config
    # ─────────────────────────────────────────────────────────────────────

    def _load_whitelist(self, agent):
        """Load main.whitelist into MAC and SSID sets."""
        try:
            wl = agent._config.get('main', {}).get('whitelist', []) or []
            for item in wl:
                s = str(item).strip()
                if not s:
                    continue
                # MAC heuristic: contains : or - and mostly hex
                if ':' in s or '-' in s:
                    normalised = self._mac_norm(s)
                    # MAC prefix match supported (e.g. "fo:od:ba")
                    if normalised and all(c in '0123456789abcdef'
                                          for c in normalised):
                        self._whitelist_macs.add(normalised)
                        continue
                # Otherwise treat as SSID
                self._whitelist_ssids.add(s)
            if self._whitelist_macs or self._whitelist_ssids:
                logging.info(
                    f'[envtune] Whitelist loaded: '
                    f'{len(self._whitelist_macs)} MACs, '
                    f'{len(self._whitelist_ssids)} SSIDs')
        except Exception as e:
            logging.debug(f'[envtune] whitelist load: {e}')

    # ─────────────────────────────────────────────────────────────────────
    # Handshake directory scan
    # ─────────────────────────────────────────────────────────────────────

    def _scan_handshake_dir(self):
        """
        Collect normalised BSSIDs with existing captures.
        Pwnagotchi filenames: <ssid>_<bssid>.pcap (bssid = last underscore).
        """
        captured = set()
        try:
            if not os.path.isdir(self.HANDSHAKE_DIR):
                return captured
            for fn in os.listdir(self.HANDSHAKE_DIR):
                if not fn.endswith(('.pcap', '.pcapng')):
                    continue
                stem  = fn.rsplit('.', 1)[0]
                parts = stem.split('_')
                if not parts:
                    continue
                mac = self._mac_norm(parts[-1])
                if len(mac) == 12 and all(c in '0123456789abcdef' for c in mac):
                    captured.add(mac)
        except Exception as e:
            logging.debug(f'[envtune] handshake dir scan: {e}')
        return captured

    # ─────────────────────────────────────────────────────────────────────
    # Bettercap sync (for wifi.* parameters that need realtime update)
    # ─────────────────────────────────────────────────────────────────────

    def _bettercap_sync(self, agent, params_changed):
        """
        Push wifi.* parameter changes to bettercap in realtime.
        Without this, ap_ttl / sta_ttl / min_rssi are silently ignored
        after pwnagotchi startup.
        """
        for param, new_val in params_changed.items():
            bcap_key = self.BETTERCAP_SYNC_MAP.get(param)
            if not bcap_key:
                continue
            try:
                agent.run(f'set {bcap_key} {new_val}')
            except Exception as e:
                logging.debug(f'[envtune] bcap sync {bcap_key}={new_val}: {e}')

    def _push_bcap_skip_list(self, agent, force=False):
        """
        Push the running set of captured BSSIDs to bettercap's
        wifi.assoc.skip and wifi.deauth.skip lists.

        Effect: bettercap stops attacking already-captured APs, freeing
        radio time for *new* targets. This is the single biggest lever
        for unique-handshake throughput once a session has been running.

        Best-effort: silently no-ops on bettercap builds that don't
        expose these properties. Coalesces — only pushes when the set
        has grown since the last push (or force=True).
        """
        if agent is None:
            return
        n = len(self._bcap_skip_macs)
        if not force and n == self._bcap_skip_pushed_count:
            return
        if not self._bcap_skip_macs:
            return
        try:
            skip_list = ','.join(sorted(self._bcap_skip_macs))
            agent.run(f'set wifi.assoc.skip {skip_list}')
            agent.run(f'set wifi.deauth.skip {skip_list}')
            self._bcap_skip_pushed_count = n
            logging.debug(f'[envtune] pushed {n} BSSIDs to bcap skip-list')
        except Exception as e:
            logging.debug(f'[envtune] bcap skip-list push: {e}')

    # ─────────────────────────────────────────────────────────────────────
    # Detect which params this fork exposes (graceful for evilsocket)
    # ─────────────────────────────────────────────────────────────────────

    def _detect_supported_params(self, agent):
        try:
            p = agent._config.get('personality', {}) or {}
            supported = {k for k in self.UCB_ARMS if k in p}
            missing   = set(self.UCB_ARMS.keys()) - supported
            if missing:
                logging.info(f'[envtune] Fork missing params: {sorted(missing)} '
                             f'— those UCB arms will be skipped')
            self._active_params = supported
        except Exception as e:
            logging.warning(f'[envtune] param detection fallback: {e}')
            self._active_params = set(self.UCB_ARMS.keys())

    # ─────────────────────────────────────────────────────────────────────
    # State persistence (async, atomic, fsync'd)
    # ─────────────────────────────────────────────────────────────────────

    def _load_state(self):
        try:
            if not os.path.exists(self.STATE_PATH):
                return
            with open(self.STATE_PATH) as f:
                st = json.load(f)

            loaded_schema = _si(st.get('schema_version', 1))

            # Sanitize loaded EMAs against EMA_CLAMP. Values from older
            # plugin versions (before the clamp safeguard) could be
            # outside the sane range — e.g. we observed reward=-8.5e15
            # in real telemetry from a one-off bad native_reward sample.
            # Drop those instead of poisoning the new run; _ema() will
            # re-seed on the next epoch.
            loaded_ema = (st.get('ema') or {})
            for k, v in loaded_ema.items():
                if k not in self.ema:
                    continue
                fv = _sf(v, default=None) if v is not None else None
                if fv is None or not math.isfinite(fv):
                    continue
                lo, hi = self.EMA_CLAMP.get(k, (-1e9, 1e9))
                if fv < lo or fv > hi:
                    logging.warning(
                        f'[envtune] dropped corrupt EMA {k}={fv} '
                        f'(outside [{lo}, {hi}])')
                    continue
                self.ema[k] = fv
            # Also reset the trend tracker — it depends on prev reward EMA
            # and would carry the corruption forward as a delta.
            self._prev_reward_ema = self.ema.get('reward')
            self._reward_trend    = 0.0
            self.lifetime_handshakes = _si(st.get('lifetime_handshakes', 0))
            self._lifetime_new_count = _si(st.get('lifetime_new_count', 0))

            # Restore captured-BSSID set. Without this, lifetime_new_count
            # could desync from disk-state (deleted pcaps) and re-counting
            # an already-known BSSID as "new" again would inflate metrics.
            # FIX: validate hex-only — a corrupted JSON entry could otherwise
            # poison the set with garbage that never normalises out (and
            # would inflate counters).
            _hex = set('0123456789abcdef')
            for m in (st.get('captured_bssids') or []):
                m_n = self._mac_norm(m)
                if len(m_n) == 12 and set(m_n).issubset(_hex):
                    self._captured_bssids.add(m_n)

            # FIX: persist cracked-BSSID set so we don't lose this knowledge
            # if the wpa-sec potfile is rotated or corrupted between runs.
            # We re-merge with the live potfile in on_loaded, so this is a
            # safety net rather than the source of truth.
            for m in (st.get('cracked_bssids') or []):
                m_n = self._mac_norm(m)
                if len(m_n) == 12 and set(m_n).issubset(_hex):
                    self._cracked_bssids.add(m_n)

            for k, v in (st.get('ch_lt') or {}).items():
                try:
                    self._ch_lt[int(k)].update(v)
                except (ValueError, TypeError):
                    continue
            for k, v in (st.get('dead_lt') or {}).items():
                try:
                    self._dead_lt[int(k)] = _si(v)
                except (ValueError, TypeError):
                    continue

            for zone_key, zdata in (st.get('gps_zones') or {}).items():
                self._gps_zones[zone_key]['hs']        = _si(zdata.get('hs', 0))
                self._gps_zones[zone_key]['attacks']   = _si(zdata.get('attacks', 0))
                self._gps_zones[zone_key]['visits']    = _si(zdata.get('visits', 0))
                self._gps_zones[zone_key]['last_seen'] = _sf(zdata.get('last_seen', 0))
                for c, n in (zdata.get('channels') or {}).items():
                    try:
                        self._gps_zones[zone_key]['channels'][int(c)] = _si(n)
                    except (ValueError, TypeError):
                        continue

            self.best_reward   = st.get('best_reward')
            self.best_settings = st.get('best_settings')

            raw_ucb = st.get('ucb_table')
            if raw_ucb:
                self._deserialise_ucb(raw_ucb, loaded_schema)

            if loaded_schema < self.STATE_SCHEMA_VERSION:
                logging.info(
                    f'[envtune] State migrated from schema v{loaded_schema} '
                    f'to v{self.STATE_SCHEMA_VERSION}')

            logging.info(
                f'[envtune] State loaded — lifetime_hs={self.lifetime_handshakes} '
                f'zones={len(self._gps_zones)} best_rwd={self.best_reward}')
        except Exception as e:
            logging.warning(f'[envtune] State load failed: {e} — starting fresh')

    def _build_state_snapshot(self):
        """Build a full state dict under the lock, return it for async write."""
        with self._state_lock:
            return {
                'schema_version':      self.STATE_SCHEMA_VERSION,
                'envtune_version':     self.__version__,
                'ema':                 dict(self.ema),
                'lifetime_handshakes': self.lifetime_handshakes,
                'lifetime_new_count':  self._lifetime_new_count,
                'captured_bssids':     sorted(self._captured_bssids),
                'cracked_bssids':      sorted(self._cracked_bssids),
                'ch_lt':   {str(k): dict(v) for k, v in self._ch_lt.items()},
                'dead_lt': {str(k): v       for k, v in self._dead_lt.items()},
                'gps_zones': {
                    zk: {
                        'hs':        z['hs'],
                        'attacks':   z['attacks'],
                        'visits':    z['visits'],
                        'last_seen': z['last_seen'],
                        'channels':  {str(c): n for c, n in z['channels'].items()},
                    }
                    for zk, z in self._gps_zones.items()
                },
                'best_reward':   self.best_reward,
                'best_settings': self.best_settings,
                'ucb_table':     self._serialise_ucb(),
                'saved_at':      time.time(),
            }

    def _save_worker(self):
        """Background thread: drain save queue, coalesce rapid requests."""
        while not self._save_stop.is_set():
            try:
                snapshot = self._save_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if snapshot is None:
                break
            # Drain additional queued snapshots — only keep the latest
            while True:
                try:
                    snapshot = self._save_queue.get_nowait()
                except queue.Empty:
                    break
                if snapshot is None:
                    self._save_stop.set()
                    return
            try:
                self._atomic_write(snapshot)
            except Exception as e:
                logging.warning(f'[envtune] async save failed: {e}')

    def _atomic_write(self, snapshot):
        """Atomic write with fsync."""
        dir_ = os.path.dirname(self.STATE_PATH) or '.'
        fd, tmp = tempfile.mkstemp(
            prefix='.envtune_', suffix='.json.tmp', dir=dir_)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(snapshot, f, separators=(',', ':'))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.STATE_PATH)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _maybe_save(self):
        self.epochs_since_save += 1
        if self.epochs_since_save >= int(self._profile['save_every_n']):
            try:
                snapshot = self._build_state_snapshot()
                # Non-blocking — drop on full queue (stale snapshots
                # matter less than blocking the main loop)
                try:
                    self._save_queue.put_nowait(snapshot)
                except queue.Full:
                    logging.debug('[envtune] save queue full — dropping snapshot')
            except Exception as e:
                logging.warning(f'[envtune] snapshot build failed: {e}')
            self.epochs_since_save = 0

    def _sync_save_now(self):
        """Force an immediate synchronous save (shutdown path)."""
        try:
            snapshot = self._build_state_snapshot()
            self._atomic_write(snapshot)
        except Exception as e:
            logging.warning(f'[envtune] sync save failed: {e}')

    def _enqueue_save(self, reason='manual'):
        """Push a snapshot to the async save queue. Drops on full queue
        (the saver coalesces; one drop is harmless)."""
        try:
            snapshot = self._build_state_snapshot()
            try:
                self._save_queue.put_nowait(snapshot)
                logging.debug(f'[envtune] save enqueued ({reason})')
            except queue.Full:
                logging.debug(f'[envtune] save queue full — drop ({reason})')
        except Exception as e:
            logging.warning(f'[envtune] enqueue save failed ({reason}): {e}')

    # ═════════════════════════════════════════════════════════════════════
    # Plugin lifecycle
    # ═════════════════════════════════════════════════════════════════════

    def on_loaded(self):
        # Merge user options into config
        try:
            user = self.options or {}
            for k, v in user.items():
                if k in self.DEFAULTS:
                    self.cfg[k] = v
        except Exception:
            pass

        # Resolve CPU profile
        chosen = self.cfg.get('cpu_profile')
        if not chosen or chosen not in CPU_PROFILES:
            hw = _detect_hardware()
            chosen = HW_DEFAULT_PROFILE.get(hw, 'balanced')
            logging.info(f'[envtune] auto-selected CPU profile "{chosen}" '
                         f'for detected hardware: {hw}')
        self._profile = dict(CPU_PROFILES[chosen])
        self._profile_name = chosen

        # Set log level
        try:
            level = self.cfg.get('log_level', 'INFO').upper()
            logging.getLogger().setLevel(getattr(logging, level, logging.INFO))
        except Exception:
            pass

        # Initialise UCB tables (cfg/profile must be ready first)
        self._init_ucb_table()

        # Load persistent state (may overwrite UCB entries with real data)
        self._load_state()

        # Apply time-of-day priors (only fills n=0 entries)
        self._apply_tod_prior()

        # Merge state-restored BSSIDs with current handshake-dir scan.
        # State has the authoritative lifetime count; disk has authoritative
        # presence — neither alone is reliable across pcap-deletes / wipes.
        self._captured_bssids |= self._scan_handshake_dir()

        # Scan wpa-sec potfile for cracked networks
        self._cracked_bssids = self._scan_cracked_potfile()

        # First-run init: if no persisted lifetime_new_count yet, seed it
        # from the existing handshake count. Otherwise we'd treat every
        # already-captured AP as "lifetime new" the next time we see it.
        if self._lifetime_new_count == 0 and len(self._captured_bssids) > 0:
            self._lifetime_new_count = len(self._captured_bssids)
            logging.info(
                f'[envtune] First run with existing handshakes — '
                f'seeding lifetime_new_count from disk '
                f'({self._lifetime_new_count} unique BSSIDs)')

        # CRITICAL: always sync prev-counters AFTER load+seed so first epoch
        # computes a clean diff. Without this, with stored count=N the
        # first-epoch diff would be N-0=N → reward spike → UCB would
        # incorrectly credit random startup parameters.
        self._lifetime_new_count_prev = self._lifetime_new_count
        self._known_aps_count_prev    = len(self._known_aps)

        # Build initial bettercap skip-list from captured + cracked BSSIDs
        # so the radio stops wasting time on duplicates we already have.
        for m in self._captured_bssids:
            fmac = _format_mac_colons(m)
            if fmac:
                self._bcap_skip_macs.add(fmac)
        for m in self._cracked_bssids:
            fmac = _format_mac_colons(m)
            if fmac:
                self._bcap_skip_macs.add(fmac)

        # Start async save thread
        self._save_thread = threading.Thread(
            target=self._save_worker, name='envtune-save', daemon=True)
        self._save_thread.start()

        logging.info(
            f'[envtune] v{self.__version__} loaded | profile={chosen} | '
            f'ucb_window={self._profile["ucb_window"]} | '
            f'lifetime_hs={self.lifetime_handshakes} | '
            f'lifetime_unique={self._lifetime_new_count} | '
            f'pre_captured={len(self._captured_bssids)} | '
            f'cracked={len(self._cracked_bssids)} | '
            f'best_reward={self.best_reward}')

    def on_ready(self, agent):
        self._agent = agent
        self._detect_supported_params(agent)
        self._load_whitelist(agent)

        # Detect GPS source
        self._gps_source = self._detect_gps_source(agent)
        if self._gps_source:
            self._gps_available = True
            logging.info(f'[envtune] GPS active via {self._gps_source}')
        else:
            logging.info('[envtune] GPS not detected — '
                         'plugin runs without zone awareness')

        # Initial bettercap sync — push current personality values to bettercap
        try:
            p = agent._config['personality']
            for param in self.BETTERCAP_SYNC_MAP:
                if param in p:
                    self._bettercap_sync(agent, {param: p[param]})
        except Exception as e:
            logging.debug(f'[envtune] initial bettercap sync: {e}')

        # Push initial captured-BSSID skip list so bettercap deprioritises
        # duplicates from the very first attack cycle of this session.
        self._push_bcap_skip_list(agent, force=True)

        if self.cfg.get('reset_history', True):
            try:
                agent._history = {}
                agent.run('wifi.recon clear')
                agent.run('wifi.clear')
                chs = agent._config['personality'].get('channels') or [1, 6, 11]
                agent.run('wifi.recon.channel %s' % ','.join(map(str, chs)))
            except Exception as e:
                logging.warning(f'[envtune] history reset: {e}')

        if agent._config.get('ai', {}).get('enabled', False):
            logging.info('[envtune] pwnagotchi AI mode is active — '
                         'envtune will be a passive observer')
        else:
            logging.info(f'[envtune] active and learning '
                         f'(tuning {len(self._active_params)} params)')

    def on_unload(self, ui):
        self._sync_save_now()
        self._save_stop.set()
        try:
            self._save_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join(timeout=2.0)
        logging.info('[envtune] unloaded — final state saved')

    def on_ui_setup(self, ui):
        self._ui = ui

    def on_ui_update(self, ui):
        # Check battery on every UI update (cheap)
        if self._ui is not None:
            self._battery_level = self._read_battery()

    # ── Mood callbacks ────────────────────────────────────────────────────
    def on_bored(self, agent):
        self._mood = 'bored'

    def on_sad(self, agent):
        self._mood = 'sad'
        # Sad pwnagotchi = persistent inactivity. Trigger exploration burst
        # to escape what is probably a stale local-optimum.
        if self._exploration_boost <= 0:
            self._exploration_boost = int(self.cfg['exploration_boost_epochs'])
            self._ucb_cache.clear()

    def on_excited(self, agent):
        self._mood = 'excited'

    def on_grateful(self, agent):
        self._mood = 'grateful'

    def on_angry(self, agent):
        self._mood = 'angry'

    # ── Free channel detection ────────────────────────────────────────────
    def on_free_channel(self, agent, channel):
        try:
            ch = _si(channel)
            if ch:
                with self._state_lock:
                    self._free_channels.append(ch)
                    self._ch_lt[ch]['free_seen'] += 1
        except Exception:
            pass

    # ── Config change callback ────────────────────────────────────────────
    def on_config_changed(self, config):
        # Re-detect supported params (web_cfg may have toggled something)
        if self._agent is not None:
            self._detect_supported_params(self._agent)

    # ═════════════════════════════════════════════════════════════════════
    # Main epoch loop — the brain
    # ═════════════════════════════════════════════════════════════════════

    def on_epoch(self, agent, epoch, epoch_data):
        # Don't fight pwnagotchi's AI if somehow active
        if agent._config.get('ai', {}).get('enabled', False):
            return

        self.epochs_seen += 1
        if self._exploration_boost > 0:
            self._exploration_boost -= 1

        try:
            # ── 1. Read raw observations ──────────────────────────────────
            try:
                raw_aps = agent.get_access_points() or []
                raw_aps = [ap for ap in raw_aps if not self._is_whitelisted(ap)]
                aps     = len(raw_aps)
            except Exception:
                raw_aps = []
                aps     = 0

            # FIX: floor at 10s, not 1s. Very short epochs (e.g. 5s) inflate
            # hs_per_min unrealistically and corrupt _recent_hpm percentiles.
            dur_secs     = max(10.0, _sf(epoch_data.get('duration_secs', 60)))
            deauths      = _si(epoch_data.get('num_deauths',          0))
            assocs       = _si(epoch_data.get('num_associations',     0))
            handshakes   = _si(epoch_data.get('num_handshakes',       0))
            missed       = _si(epoch_data.get('missed_interactions',  0))
            blind_for    = _si(epoch_data.get('blind_for_epochs',     0))
            active_for   = _si(epoch_data.get('active_for_epochs',    0))
            inactive_for = _si(epoch_data.get('inactive_for_epochs',  0))
            num_hops     = _si(epoch_data.get('num_hops',             0))
            temperature  = _sf(epoch_data.get('temperature',          40.0))
            cpu_load     = _sf(epoch_data.get('cpu_load',             0.0))
            native_rwd   = _sf(epoch_data.get('reward', 0.0))
            ep_total     = max(1, _si(epoch_data.get('epoch', epoch)) or epoch or 1)
            # Mood counters from pwnagotchi's own epoch tracker. Original AI
            # gated these at 5 epochs to avoid penalising warm-up; we keep
            # the same threshold.
            bored_for    = _si(epoch_data.get('bored_for_epochs',     0))
            sad_for      = _si(epoch_data.get('sad_for_epochs',       0))

            interactions = deauths + assocs
            hs_rate      = handshakes / interactions if interactions > 0 else 0.0
            missed_rate  = missed    / interactions if interactions > 0 else 0.0
            hs_per_min   = handshakes / (dur_secs / 60.0)

            # FIX: use pwnagotchi's own epoch counter for ratios
            active_ratio   = active_for   / ep_total
            inactive_ratio = inactive_for / ep_total
            tot_ch         = max(len(self._ch_lt), 14)
            hops_ratio     = min(1.0, num_hops / max(1, tot_ch))

            # FIX: lifetime-new captures (NOT session-new). The whole point
            # of EnvTune is maximising captures of networks we have NEVER
            # seen before across all sessions. We track this via the
            # _captured_bssids set (loaded from /root/handshakes/ + grown
            # in on_handshake). on_handshake increments _lifetime_new_count
            # whenever a brand-new BSSID is captured.
            lifetime_new_this_epoch = (
                self._lifetime_new_count - getattr(
                    self, '_lifetime_new_count_prev', 0))
            lifetime_new_this_epoch = max(0, lifetime_new_this_epoch)
            self._lifetime_new_count_prev = self._lifetime_new_count

            # New APs discovered this epoch (not necessarily captured —
            # exploration value, even when no handshake yet)
            current_ap_count = len(self._known_aps)
            new_aps_seen = max(0, current_ap_count - getattr(
                self, '_known_aps_count_prev', 0))
            self._known_aps_count_prev = current_ap_count

            # FIX: snapshot _known_aps under lock once per epoch. All later
            # iterations in on_epoch use the snapshot, avoiding races with
            # on_wifi_update / on_handshake / on_association mutations.
            with self._state_lock:
                aps_items_snap  = list(self._known_aps.items())
                aps_values_snap = [v for _, v in aps_items_snap]

            # "Underlying work" proxy — did we DO things this epoch even
            # if no handshake came out? Assoc/deauth attempts on uncaptured
            # APs (vs wasted on already-captured) count as productive work.
            # Avoids the trap where 0-handshake epochs all look identical.
            uncaptured_attacks = sum(
                ap.get('AT_attacks', 0) for ap in aps_values_snap
                if not ap.get('AT_already_captured', False)
            )
            attack_efficiency_proxy = (
                min(1.0, interactions / 10.0) if uncaptured_attacks > 0 else 0.0
            )

            # ── 2. Save pre-update aps_ema for nexmon crash check ─────────
            self._prev_aps_ema = self.ema.get('aps')

            # ── 3. Update GPS fix and mobility ────────────────────────────
            if self._gps_available:
                fix = self._read_gps(agent)
                if fix is not None:
                    self._gps_last_fix = fix
                self._update_gps_zone()
            self._current_mobility = self._compute_mobility()

            # ── 4. Update EMAs ────────────────────────────────────────────
            aps_ema  = self._ema('aps',            aps)
            hs_ema   = self._ema('hs_rate',        hs_rate)
            r_ema    = self._ema('reward',         native_rwd)
            mi_ema   = self._ema('missed_rate',    missed_rate)
            _        = self._ema('hs_per_min',     hs_per_min)
            _        = self._ema('active_ratio',   active_ratio)
            _        = self._ema('inactive_ratio', inactive_ratio)
            _        = self._ema('hops_per_epoch', num_hops)
            t_ema    = self._ema('temperature',    temperature)
            _        = self._ema('cpu_load',       cpu_load)
            if self._gps_last_fix is not None:
                _ = self._ema('speed', self._gps_last_fix.get('speed', 0))

            if self._prev_reward_ema is not None:
                self._reward_trend = r_ema - self._prev_reward_ema
            self._prev_reward_ema = r_ema

            # ── 5. Compute custom reward ──────────────────────────────────
            blind_ratio = blind_for / ep_total
            bored_ratio = (bored_for / ep_total) if bored_for >= 5 else 0.0
            sad_ratio   = (sad_for   / ep_total) if sad_for   >= 5 else 0.0
            custom_rwd = self._custom_reward(
                handshakes, hs_rate, missed_rate, native_rwd, dur_secs,
                lifetime_new_this_epoch, active_ratio, inactive_ratio, hops_ratio,
                new_aps_seen, attack_efficiency_proxy, interactions,
                blind_ratio=blind_ratio, bored_ratio=bored_ratio,
                sad_ratio=sad_ratio)

            # ── 6. Nexmon crash detection ─────────────────────────────────
            if self._check_nexmon_crash(aps, interactions):
                logging.warning('[envtune] nexmon crash suspected — '
                                'aggressive throttle')
                p = agent._config['personality']
                if 'throttle_d' in self._active_params:
                    p['throttle_d'] = min(1.2, _sf(p.get('throttle_d', 0.9)) + 0.3)
                if 'throttle_a' in self._active_params:
                    p['throttle_a'] = min(1.0, _sf(p.get('throttle_a', 0.4)) + 0.2)
                p['max_interactions'] = max(2, _si(p.get('max_interactions', 3)) - 1)
                self._schedule_channels(agent)
                self._reset_decision_buffer()
                self._maybe_save()
                return

            # ── 7. Thermal safety ─────────────────────────────────────────
            if t_ema > 0:
                self._apply_thermal_throttle(agent, t_ema)

            # ── 8. Location change ────────────────────────────────────────
            fp = self._compute_location_fp(raw_aps)
            if self._check_location_change(fp):
                boost = int(self.cfg['exploration_boost_epochs']) * 2
                self._exploration_boost = boost
                self._dead_session.clear()
                self._free_channels.clear()
                self._ucb_cache.clear()
                # FIX: stale decisions in the buffer were taken in the
                # previous environment — attributing rewards from the new
                # environment to them corrupts UCB stats. But discarding
                # them entirely means UCB never learns *anything* from arms
                # explored just before a move, which under-tests them. Give
                # each pending decision a neutral 0.5 credit so the visit
                # count grows but the mean is centred — UCB will still
                # explore them again, just not pessimistically.
                neutral = 0.5
                for _ep, old_state, old_params in list(self._decision_buffer):
                    for param, val in old_params.items():
                        self._ucb_update(param, old_state, val, neutral)
                self._reset_decision_buffer()
                logging.info(f'[envtune] location change → '
                             f'{boost}-ep exploration boost '
                             f'(neutral-credited buffered decisions)')

            # ── 9. Attribute delayed reward to earlier decision ───────────
            # Adaptive reward_delay: in dense AP environments, parameter
            # changes show in the next 1-2 epochs; in sparse environments,
            # they take longer to manifest (slow scan/handshake cadence).
            base_delay = int(self.cfg['reward_delay'])
            if aps_ema >= 25:
                delay = max(2, base_delay - 1)
            elif aps_ema <= 5:
                delay = base_delay + 1
            else:
                delay = base_delay
            if len(self._decision_buffer) >= delay:
                old_ep, old_state, old_params = list(self._decision_buffer)[-delay]
                for param, val in old_params.items():
                    self._ucb_update(param, old_state, val, custom_rwd)

            # ── 10. Stagnation check ──────────────────────────────────────
            self._check_stagnation(custom_rwd)

            # ── 10b. Saturation-aware exploration boost ───────────────────
            # If we've captured most of the APs visible in this location,
            # there's nothing more to capture without moving — push
            # exploration so we test scan params that might surface the
            # last few hidden / weak APs (deeper min_rssi, longer recon).
            now_mono = time.monotonic()
            with self._state_lock:
                visible = 0
                cap_in_view = 0
                for ap in self._known_aps.values():
                    if ap.get('AT_cracked', False):
                        continue
                    if (now_mono - ap.get('AT_lastseen', 0)) > 90:
                        continue
                    visible += 1
                    if ap.get('AT_already_captured', False):
                        cap_in_view += 1
            if visible >= 8 and cap_in_view / max(1, visible) > 0.80:
                if self._exploration_boost <= 0:
                    self._exploration_boost = int(
                        self.cfg['exploration_boost_epochs'])
                    logging.info(
                        f'[envtune] saturation '
                        f'({cap_in_view}/{visible} captured nearby) → '
                        f'{self._exploration_boost}-ep exploration boost')

            # ── 11. Best-settings tracking ────────────────────────────────
            if self.best_reward is None or custom_rwd > self.best_reward + 0.03:
                self.best_reward = custom_rwd
                pdict = agent._config['personality']
                self.best_settings = {k: pdict.get(k) for k in self._active_params}

            # ── 12. Blind-panic handling ──────────────────────────────────
            p = agent._config['personality']
            if blind_for >= int(self.cfg['blind_panic_epochs']):
                if self._blind_recovery == 0:
                    self._blind_saved_params = {
                        k: p.get(k) for k in self._active_params}
                    logging.warning(f'[envtune] BLIND PANIC '
                                    f'(blind_for={blind_for})')
                p['min_rssi']         = self.BOUNDS['min_rssi'][0]
                p['recon_time']       = self.BOUNDS['recon_time'][1]
                p['hop_recon_time']   = 8
                # If thermal throttle was already in effect this epoch
                # (step 7 ran before us), keep its lower max_interactions
                # rather than reset to 3 — overheating + blind is the worst
                # combo and we should not loosen the thermal lid.
                if self._thermal_throttle:
                    p['max_interactions'] = min(
                        3, _si(p.get('max_interactions', 3)))
                else:
                    p['max_interactions'] = 3
                if 'throttle_a' in self._active_params:
                    # Slow association attempts to give the radio room to
                    # finish a scan/recover from firmware indigestion.
                    # Keep the more conservative of (blind=0.4, thermal-set).
                    p['throttle_a'] = max(0.4, _sf(p.get('throttle_a', 0.4)))
                if 'throttle_d' in self._active_params:
                    p['throttle_d'] = max(0.9, _sf(p.get('throttle_d', 0.9)))
                self._bettercap_sync(agent, {
                    'min_rssi': p['min_rssi'],
                })
                self._blind_recovery = int(self.cfg['blind_recovery_steps'])
                self._schedule_channels(agent)
                self._reset_decision_buffer()
                self._maybe_save()
                return

            # Gradual recovery from blind panic
            if self._blind_recovery > 0 and self._blind_saved_params:
                self._blind_recovery -= 1
                synced = {}
                for param, saved_val in self._blind_saved_params.items():
                    if saved_val is None or param not in self.UCB_ARMS:
                        continue
                    arms = sorted(self.UCB_ARMS[param])
                    if not arms:
                        continue
                    cur_val = _sf(p.get(param, saved_val))
                    try:
                        ci   = arms.index(min(arms, key=lambda a: abs(a - cur_val)))
                        ti   = arms.index(min(arms, key=lambda a: abs(a - _sf(saved_val))))
                        step = 1 if ti > ci else (-1 if ti < ci else 0)
                        new_val = arms[max(0, min(len(arms) - 1, ci + step))]
                        p[param] = new_val
                        if param in self.BETTERCAP_SYNC_MAP:
                            synced[param] = new_val
                    except (ValueError, IndexError):
                        p[param] = saved_val
                if synced:
                    self._bettercap_sync(agent, synced)
                if self._blind_recovery == 0:
                    self._blind_saved_params = None
                self._schedule_channels(agent)
                self._reset_decision_buffer()
                self._maybe_save()
                return

            # ── 13. Warmup: just observe ──────────────────────────────────
            if self.epochs_seen < int(self.cfg['warmup_epochs']):
                self._schedule_channels(agent)
                self._maybe_save()
                return

            # ── 14. Skip tuning during thermal throttle ───────────────────
            if self._thermal_throttle:
                self._schedule_channels(agent)
                self._reset_decision_buffer()
                self._maybe_save()
                return

            # ── 15. Compute environment state ─────────────────────────────
            state = self._compute_state(aps_ema)

            # ── 16. UCB select arms for active parameters ─────────────────
            chosen = {
                param: self._ucb_select(param, state)
                for param in self.UCB_ARMS
                if param in self._active_params
            }

            # ── 17. Client-aware override ─────────────────────────────────
            recency_limit = int(self.cfg['client_recency_epochs'])
            total_fresh_clients = sum(
                ap.get('AT_clients', 0)
                for ap in aps_values_snap
                if (not ap.get('AT_already_captured', False)
                    and ap.get('AT_cooldown_until', 0) <= self.epochs_seen
                    and (self.epochs_seen - ap.get('AT_client_epoch', -99))
                        <= recency_limit)
            )
            # FIX: also drop max_interactions in genuinely sparse environments
            # (few APs total) — interactions threshold alone misses the "small
            # cafe with 3 strong APs" case where we should still favor PMKID.
            if (total_fresh_clients == 0
                    and (interactions >= 3 or aps_ema < 5)):
                # No clients → focus on PMKID (assoc), reduce deauth aggression
                chosen['max_interactions'] = min(
                    _si(chosen.get('max_interactions', 3)), 2)
            elif total_fresh_clients >= 5:
                chosen['max_interactions'] = max(
                    _si(chosen.get('max_interactions', 3)), 4)

            # ── 18. PMF detection ─────────────────────────────────────────
            # FIX: detection is one-way. Once AT_pmf_detected=True we never
            # try that AP again, but firmware/client-cap can change.
            # Re-evaluate every 200 epochs after detection: if a fresh
            # client appears AND we are well within range, allow one more
            # attempt by clearing the flag (and resetting attack counter).
            pmf_thr = int(self.cfg['pmf_attack_threshold'])
            for apID, ap in aps_items_snap:
                if (ap.get('AT_attacks', 0) >= pmf_thr
                        and ap.get('AT_handshake', 0) == 0
                        and _sf(ap.get('rssi', -85)) > -72):
                    ap['AT_pmf_detected'] = True
                    ap['AT_pmf_detected_ep'] = self.epochs_seen
                elif ap.get('AT_pmf_detected', False):
                    pmf_ep   = ap.get('AT_pmf_detected_ep', 0)
                    age      = self.epochs_seen - pmf_ep
                    has_fresh = (
                        ap.get('AT_clients', 0) > 0
                        and (self.epochs_seen
                             - ap.get('AT_client_epoch', -99))
                            <= recency_limit)
                    if (age >= 200
                            and has_fresh
                            and _sf(ap.get('rssi', -85)) > -65):
                        ap['AT_pmf_detected'] = False
                        ap['AT_attacks']      = 0
                        ap['AT_missed']       = 0

            # ── 19. Sanity check parameter coupling ───────────────────────
            chosen = self._sanity_check(chosen)

            # ── 20. Apply parameters ──────────────────────────────────────
            sync_needed = {}
            for param, val in chosen.items():
                old = p.get(param)
                p[param] = val
                if param in self.BETTERCAP_SYNC_MAP and old != val:
                    sync_needed[param] = val
            if sync_needed:
                self._bettercap_sync(agent, sync_needed)

            # Record decision for delayed reward attribution
            self._decision_buffer.append((epoch, state, dict(chosen)))

            # ── 21. AP cooldown & efficiency update ───────────────────────
            cd_atk    = int(self.cfg['ap_cooldown_attacks'])
            cd_short  = int(self.cfg['ap_cooldown_short'])
            cd_long   = int(self.cfg['ap_cooldown_long'])
            miss_cd   = int(self.cfg['missed_cooldown_threshold'])
            for apID, ap in aps_items_snap:
                atk = ap.get('AT_attacks', 0)
                hs  = ap.get('AT_handshake', 0)
                ap['AT_efficiency'] = hs / atk if atk > 0 else 0.0

                # ANTI-OVERCAPTURE: if we already have a handshake for this
                # AP (in /root/handshakes/), keep it on permanent rolling
                # cooldown. We can't stop pwnagotchi's main loop from going
                # for it, but we can ensure our channel scoring and
                # proactive logic ignores it. Long cooldown is deliberate:
                # prevents repeat attacks all session.
                if ap.get('AT_already_captured', False):
                    if ap.get('AT_cooldown_until', 0) <= self.epochs_seen:
                        ap['AT_cooldown_until'] = self.epochs_seen + cd_long * 4
                    continue

                # Standard cooldown on attacks-without-HS
                if (atk >= cd_atk and hs == 0
                        and ap.get('AT_cooldown_until', 0) <= self.epochs_seen):
                    cd_dur = cd_long if atk >= cd_atk * 2 else cd_short
                    ap['AT_cooldown_until'] = self.epochs_seen + cd_dur
                    continue

                # Early cooldown on excessive missed-interaction count
                if (ap.get('AT_missed', 0) >= miss_cd
                        and ap.get('AT_cooldown_until', 0) <= self.epochs_seen):
                    ap['AT_cooldown_until'] = self.epochs_seen + cd_short
                    ap['AT_missed'] = 0  # reset counter post-cooldown

            # ── 22. Channel wasted-attack tracking ────────────────────────
            if interactions > 0 and handshakes == 0:
                with self._state_lock:
                    for ch in self._active_channels:
                        self._ch_lt[ch]['wasted'] += 1

            # ── 23. Channel scheduling ────────────────────────────────────
            self._schedule_channels(agent)

            # FIX: push grown skip-list to bettercap so wifi.assoc/deauth
            # don't waste airtime on already-captured BSSIDs.
            self._push_bcap_skip_list(agent)

            # ── 24. Proactive attacks for high-value targets (opt-in) ─────
            if (self._profile['enable_proactive']
                    and self.cfg.get('opportunistic_overrides', True)
                    and self.epochs_seen - self._last_proactive_ep
                        >= int(self.cfg['proactive_gap_epochs'])
                    and not self._thermal_throttle):
                self._maybe_proactive_attack(agent)

            # ── 25. GPS zone bookkeeping ──────────────────────────────────
            if self._current_zone is not None:
                if interactions > 0:
                    self._gps_zones[self._current_zone]['attacks'] += interactions

            # ── 26. Compact INFO log line ─────────────────────────────────
            top_ch = sorted(self._ch_lt.items(),
                            key=lambda x: -x[1]['hs'])[:3]
            top_s  = ','.join(f'{c}:{d["hs"]}' for c, d in top_ch) or 'none'
            zone_s = self._current_zone or '-'
            logging.info(
                f'[envtune] ep={epoch} st={state} mood={self._mood} '
                f'aps={aps_ema:.0f} hs_rt={hs_ema:.2f} '
                f'hpm={self.ema["hs_per_min"]:.2f} miss={mi_ema:.2f} '
                f'rwd={custom_rwd:.2f} t={t_ema:.0f}C '
                f'unique_lifetime={self._lifetime_new_count} '
                f'(+{lifetime_new_this_epoch} this ep) '
                f'top={top_s} zone={zone_s} mob={self._current_mobility}')

            # ── 27. Verbose DEBUG dump ────────────────────────────────────
            logging.debug(f'[envtune] params={chosen} expl={self._exploration_boost} '
                          f'fresh_clients={total_fresh_clients}')
            if self._last_reward_breakdown:
                # Compact one-line component log so operators can see WHY
                # reward landed where it did. Sorted by absolute weight so
                # dominant terms come first.
                items = sorted(
                    self._last_reward_breakdown.items(),
                    key=lambda kv: -abs(kv[1]))
                comps = ' '.join(f'{k}={v:+.3f}' for k, v in items)
                logging.debug(f'[envtune] reward_components: {comps}')

            # ── 28. Periodic wpa-sec potfile rescan ───────────────────────
            # External tool (cron / wpa-sec.py) appends to the potfile
            # asynchronously; if we never rescan, freshly cracked networks
            # keep being targeted long after we know the password.
            if (self.cfg.get('enable_wpasec_feedback', True)
                    and self.epochs_seen
                    and self.epochs_seen % int(
                        self.cfg.get('potfile_rescan_every_n', 100)) == 0):
                try:
                    cracked = self._scan_cracked_potfile()
                    if cracked:
                        with self._state_lock:
                            added = len(cracked - self._cracked_bssids)
                            self._cracked_bssids = cracked
                            if added:
                                for ap in self._known_aps.values():
                                    if self._mac_norm(ap.get('mac', '')) in cracked:
                                        ap['AT_cracked'] = True
                        if added:
                            logging.info(f'[envtune] potfile rescan: '
                                         f'+{added} cracked BSSIDs '
                                         f'({len(cracked)} total)')
                except Exception as e:
                    logging.debug(f'[envtune] potfile rescan: {e}')

            # ── 29. Handshake-dir rescan watchdog ─────────────────────────
            # If something external (wpa-sec sync, manual copy, another
            # plugin) drops a .pcap into HANDSHAKE_DIR, we won't notice
            # until a restart. Periodically diff the directory against
            # our in-memory _captured_bssids and adopt anything new — so
            # the priority loop and skip-list stay accurate.
            if (self.epochs_seen
                    and self.epochs_seen % int(
                        self.cfg.get('handshake_rescan_every_n', 200)) == 0):
                try:
                    fs_set = self._scan_handshake_dir()
                    with self._state_lock:
                        new_macs = fs_set - self._captured_bssids
                        if new_macs:
                            self._captured_bssids |= new_macs
                            self._lifetime_new_count = max(
                                self._lifetime_new_count,
                                len(self._captured_bssids))
                            for m in new_macs:
                                colon = _format_mac_colons(m)
                                if colon:
                                    self._bcap_skip_macs.add(colon)
                    if new_macs:
                        logging.info(
                            f'[envtune] handshake-dir watchdog: '
                            f'+{len(new_macs)} BSSIDs adopted from disk')
                        try:
                            self._push_bcap_skip_list(agent)
                        except Exception:
                            pass
                except Exception as e:
                    logging.debug(f'[envtune] handshake-dir watchdog: {e}')

            self._maybe_save()

        except Exception as e:
            logging.exception(f'[envtune] on_epoch: {e}')

    def _reset_decision_buffer(self):
        """Clear delayed-reward queue when we skip the UCB select path."""
        self._decision_buffer.clear()

    def _maybe_proactive_attack(self, agent):
        """
        Proactively trigger wifi.assoc on a single high-value target.
        Only if profile permits it AND there's a clearly valuable AP.
        Conservative: max 1 per N epochs, opt-in via config flag.

        Strict filters (we only want to attack worthwhile targets):
          - Not already captured (would be wasted reward)
          - Not in cooldown (we already tried recently)
          - Not PMF-detected (waste of breath)
          - Not in wpa-sec cracked set (we know the password)
          - Hidden hostname: only with strict RSSI+clients gate
          - Strong enough RSSI
          - MAC validates as real (not a malformed bcap entry)
        """
        try:
            # FIX: snapshot under lock to avoid race with on_wifi_update.
            with self._state_lock:
                aps_snap     = list(self._known_aps.items())
                cracked_snap = set(self._cracked_bssids)

            best_ap    = None
            best_score = 0.0
            for apID, ap in aps_snap:
                if ap.get('AT_already_captured', False):
                    continue
                if ap.get('AT_cooldown_until', 0) > self.epochs_seen:
                    continue
                if ap.get('AT_pmf_detected', False):
                    continue
                # FIX: skip APs whose password we already cracked via wpa-sec.
                # No reward for re-capturing networks we've already broken.
                mac_n = self._mac_norm(ap.get('mac', ''))
                if mac_n and mac_n in cracked_snap:
                    continue
                rssi    = _sf(ap.get('rssi', -85))
                clients = ap.get('AT_clients', 0)
                # FIX: hidden APs aren't useless for PMKID — bettercap can
                # still elicit an assoc frame. Allow them, but require a
                # stronger gate: very close RSSI AND active clients.
                hostname = str(ap.get('hostname', '')).strip()
                is_hidden = (
                    not hostname
                    or hostname == '<hidden>'
                    or apID.startswith('hidden-'))
                if is_hidden:
                    if rssi < -60 or clients == 0:
                        continue
                if rssi < self.cfg['proactive_min_rssi']:
                    continue
                # FIX: validate MAC syntactically before sending to bcap —
                # malformed entries would cause the agent.run command to
                # silently fail or, worse, parse wrong.
                mac = ap.get('mac', '')
                if not _is_valid_mac(mac):
                    continue
                # Score: rssi + client count
                score = (rssi + 90) + clients * 5
                if score > best_score:
                    best_score = score
                    best_ap = ap

            if best_ap is None:
                return
            mac = best_ap.get('mac')
            # Proactive PMKID grab via wifi.assoc — bettercap sends an
            # association frame, AP may leak PMKID without needing a client.
            # We do NOT do proactive deauth here: deauth requires a client
            # mac and must be timed against a real client connection, which
            # bettercap's main loop handles better than we can.
            agent.run('wifi.assoc %s' % mac)
            self._last_proactive_ep = self.epochs_seen
            with self._state_lock:
                if best_ap is self._known_aps.get(self._ap_id(best_ap)):
                    best_ap['AT_lastattack_ep'] = self.epochs_seen
            logging.debug(f'[envtune] proactive assoc → {mac}')
        except Exception as e:
            logging.debug(f'[envtune] proactive: {e}')

    # ═════════════════════════════════════════════════════════════════════
    # Event callbacks
    # ═════════════════════════════════════════════════════════════════════

    def on_handshake(self, agent, filename, access_point, client_station):
        """Record a captured handshake — the only thing we truly care about."""
        try:
            ch = 0
            mac_n = ''
            apID = None
            passive = False
            is_lifetime_new = False  # default — overwritten below if applicable

            if isinstance(access_point, dict):
                ch    = _si(access_point.get('channel', 0))
                apID  = self._ap_id(access_point)
                mac_n = self._mac_norm(access_point.get('mac', ''))
                self._mark_ap_seen(access_point, 'handshake')
                with self._state_lock:
                    if apID in self._known_aps:
                        ap = self._known_aps[apID]
                        ap['AT_handshake'] = ap.get('AT_handshake', 0) + 1
                        ap['AT_already_captured'] = True
                        ap['AT_pmkid_success'] = True
                        # Passive capture detection: 0 attacks = pure luck
                        if ap.get('AT_attacks', 0) == 0:
                            passive = True

            with self._state_lock:
                self.lifetime_handshakes += 1
                # CRITICAL: distinguish lifetime-new vs. duplicate captures.
                # _captured_bssids is loaded from /root/handshakes/ at start
                # AND maintained across sessions via state save. So if mac_n
                # is NOT in there yet, this is a brand-new capture.
                if mac_n and mac_n not in self._captured_bssids:
                    self._lifetime_new_count += 1
                    is_lifetime_new = True
                else:
                    is_lifetime_new = False
                if mac_n:
                    self._captured_bssids.add(mac_n)
                    self._session_hs_bssids.add(mac_n)
                    # FIX: feed bettercap skip-list so duplicate captures
                    # are pre-empted in the next epoch's sync. Coalesced
                    # via _push_bcap_skip_list (only sends when set grows).
                    colon_mac = _format_mac_colons(mac_n)
                    if colon_mac:
                        self._bcap_skip_macs.add(colon_mac)
                if apID:
                    self._captured_aps.add(apID)
                if ch:
                    self._inc_ch('Handshakes', ch)
                    self._ch_lt[ch]['hs'] += 1
                    if passive:
                        self._ch_lt[ch]['passive_hs'] += 1

                # GPS zone credit
                if self._current_zone is not None:
                    self._gps_zones[self._current_zone]['hs'] += 1
                    if ch:
                        self._gps_zones[self._current_zone]['channels'][ch] += 1

            self.last_shake = {
                'time': time.time(),
                'ap':   access_point,
                'cl':   client_station,
                'passive': passive,
                'lifetime_new': is_lifetime_new,
            }
            tags = []
            if is_lifetime_new:
                tags.append('🆕NEW')
            else:
                tags.append('dup')
            tags.append('PASSIVE' if passive else 'ACTIVE')
            logging.info(f'[envtune] handshake [{" ".join(tags)}] ch={ch} '
                         f'lifetime={self.lifetime_handshakes} '
                         f'unique_lifetime={self._lifetime_new_count}')

            # FIX: push the new BSSID to bettercap's skip-list immediately
            # rather than waiting for the next epoch. Otherwise bettercap
            # may re-deauth a freshly captured AP for up to 30 s, wasting
            # radio time we could spend on still-uncaptured targets. This
            # is rate-limited via the coalescing logic in _push_bcap_skip_list.
            if is_lifetime_new and agent is not None:
                try:
                    self._push_bcap_skip_list(agent)
                except Exception as e:
                    logging.debug(f'[envtune] immediate skip push: {e}')
        except Exception as e:
            logging.debug(f'[envtune] on_handshake: {e}')

    def on_association(self, agent, access_point):
        try:
            ch   = _si(access_point.get('channel', 0))
            apID = self._ap_id(access_point)
            self._mark_ap_seen(access_point, 'assoc')
            with self._state_lock:
                self._inc_ch('Associations', ch)
                self._ch_lt[ch]['assocs'] += 1
                if apID in self._known_aps:
                    ap = self._known_aps[apID]
                    ap['AT_attacks'] = ap.get('AT_attacks', 0) + 1
                    ap['AT_lastattack_ep'] = self.epochs_seen
        except Exception as e:
            logging.debug(f'[envtune] on_association: {e}')

    def on_deauthentication(self, agent, access_point, client_station):
        try:
            ch   = _si(access_point.get('channel', 0))
            apID = self._ap_id(access_point)
            self._mark_ap_seen(access_point, 'deauth')
            with self._state_lock:
                self._inc_ch('Deauths', ch)
                self._ch_lt[ch]['deauths'] += 1
                if apID in self._known_aps:
                    ap = self._known_aps[apID]
                    ap['AT_attacks'] = ap.get('AT_attacks', 0) + 1
                    ap['AT_lastattack_ep'] = self.epochs_seen
        except Exception as e:
            logging.debug(f'[envtune] on_deauthentication: {e}')

    def on_wifi_update(self, agent, access_points):
        try:
            # FIX: 'Current APs' counter must be decremented symmetrically
            # when an AP transitions visible→invisible. Previously we set
            # AT_visible=False without dec'ing the channel counter.
            # FIX: snapshot _known_aps via list() to avoid 'dict changed size'
            # under RLock re-entry from _mark_ap_seen / evict.
            # FIX: all _ch_lt, _unscanned_channels, _dead_session, _dead_lt
            # mutations now under a single lock — these are concurrently
            # read by _schedule_channels and _ch_score from on_epoch.
            with self._state_lock:
                for ap in list(self._known_aps.values()):
                    if ap.get('AT_visible', False):
                        ap_ch = _si(ap.get('channel', 0))
                        if ap_ch:
                            self._inc_ch('Current APs', ap_ch, -1)
                    ap['AT_visible'] = False

                active      = []
                visited_chs = set()
                for ap in access_points:
                    if self._is_whitelisted(ap):
                        continue
                    self._mark_ap_seen(ap, 'wifi_update')
                    ch = _si(ap.get('channel', 0))
                    if ch <= 0:
                        continue
                    if ch not in active:
                        active.append(ch)
                        if ch in self._unscanned_channels:
                            self._unscanned_channels.remove(ch)
                        self._dead_session[ch] = 0
                    if ch not in visited_chs:
                        self._ch_lt[ch]['visits'] += 1
                        visited_chs.add(ch)

                # Dead-channel session counter
                for ch in list(self._dead_session):
                    if ch not in active:
                        self._dead_session[ch] += 1
                        if (self._dead_session[ch]
                                > int(self.cfg['dead_channel_cooldown']) * 4):
                            self._dead_lt[ch] = self._dead_lt.get(ch, 0) + 1

                self._active_channels = active
        except Exception as e:
            logging.exception(f'[envtune] on_wifi_update: {e}')

    def on_bcap_wifi_ap_new(self, agent, event):
        try:
            self._mark_ap_seen(event.get('data', {}))
        except Exception:
            pass

    def on_bcap_wifi_ap_lost(self, agent, event):
        try:
            ap   = event.get('data', {})
            apID = self._ap_id(ap)
            ch   = _si(ap.get('channel', 0))
            with self._state_lock:
                if (apID in self._known_aps
                        and self._known_aps[apID].get('AT_visible', False)):
                    self._known_aps[apID]['AT_visible'] = False
                    self._inc_ch('Current APs', ch, -1)
        except Exception:
            pass

    def on_bcap_wifi_client_new(self, agent, event):
        try:
            data = event.get('data', {}) or {}
            ap   = data.get('AP', {}) or {}
            ch   = _si(ap.get('channel', 0))
            if not ch:
                return
            apID = self._ap_id(ap)
            with self._state_lock:
                self._inc_ch('Clients', ch)
                self._ch_lt[ch]['clients'] += 1
                if apID in self._known_aps:
                    self._known_aps[apID]['AT_clients'] = (
                        self._known_aps[apID].get('AT_clients', 0) + 1)
                    self._known_aps[apID]['AT_client_epoch'] = self.epochs_seen
            # Opportunistic channel override
            if (self.cfg.get('opportunistic_overrides', True)
                    and ch not in self._active_channels
                    and self.epochs_seen - self._last_override_ep
                        >= int(self.cfg['opportunistic_min_gap'])):
                try:
                    current = list(
                        agent._config['personality'].get('channels', []))
                    if ch not in current:
                        current.insert(0, ch)
                    agent.run('wifi.recon.channel %s' %
                              ','.join(map(str, current)))
                    self._last_override_ep = self.epochs_seen
                    logging.debug(f'[envtune] opportunistic override → ch {ch}')
                except Exception:
                    pass
        except Exception as e:
            logging.debug(f'[envtune] on_bcap_wifi_client_new: {e}')

    def on_bcap_wifi_client_lost(self, agent, event):
        try:
            data = event.get('data', {}) or {}
            ap   = data.get('AP', {}) or {}
            apID = self._ap_id(ap)
            with self._state_lock:
                if apID in self._known_aps:
                    cur = self._known_aps[apID].get('AT_clients', 0)
                    self._known_aps[apID]['AT_clients'] = max(0, cur - 1)
        except Exception:
            pass

    # Track missed interactions per AP for early cooldown signal
    def on_bcap_wifi_assoc(self, agent, event):
        # bettercap fires this on EACH association attempt; count missed ones
        # by comparing with our own attack counter delta later. For now,
        # increment attempts; missed is counted via epoch_data.missed_interactions
        pass

    # ═════════════════════════════════════════════════════════════════════
    # Web UI (/plugins/envtune/)
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _html_response(body, status=200):
        # Bypass Jinja: pwnagotchi UI may pass attacker-controlled SSIDs through
        # this method, so we never let `{{ }}` reach a template engine.
        resp = make_response(body, status)
        resp.headers['Content-Type'] = 'text/html; charset=utf-8'
        resp.headers['X-Content-Type-Options'] = 'nosniff'
        resp.headers['Cache-Control'] = 'no-store'
        return resp

    def _plugin_base(self):
        """Absolute URL prefix where this plugin is mounted in pwnagotchi's
        webserver. Pwnagotchi mounts at /plugins/<class_name_lowercase>/.
        Using absolute paths fixes the bug where a relative form action
        like `force-save` resolves against `/plugins/` (instead of
        `/plugins/envtune/`) when the user visits the dashboard URL
        without a trailing slash."""
        return '/plugins/' + type(self).__name__.lower() + '/'

    def on_webhook(self, path, request):
        if not self._agent:
            return self._html_response(
                '<!DOCTYPE html><html><body><h1>EnvTune not ready yet</h1>'
                '</body></html>', status=503)
        try:
            method = (request.method if request is not None else 'GET').upper()

            # POST actions (force-save / reset-stagnation / rescan-potfile)
            if method == 'POST':
                return self._handle_post(path, request)

            # Sub-paths for data export
            if path == 'export':
                return self._endpoint_export()
            if path == 'metrics':
                return self._endpoint_metrics()
            if path == 'zones':
                return self._endpoint_zones()

            # Main HTML dashboard — every dynamic value goes through html.escape
            # in its helper, and the whole document is returned as a raw HTML
            # response (no Jinja evaluation).
            version = html.escape(str(self.__version__))
            profile = html.escape(str(self._profile_name))
            gps_src = html.escape(str(self._gps_source or 'off'))
            mood = html.escape(str(self._mood))
            mobility = html.escape(str(self._current_mobility))
            base = html.escape(self._plugin_base())

            parts = [
                '<!DOCTYPE html><html><head>',
                f'<title>EnvTune v{version}</title>',
                # <base> makes ALL relative URLs (links, forms, redirects)
                # resolve against the plugin mount point, regardless of
                # whether the visitor's URL had a trailing slash.
                f'<base href="{base}">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                f'<style>{self._ui_css()}</style></head><body>',
                f'<h1>⚡ EnvTune v{version}</h1>',
                '<p class="subtitle">',
                f'profile=<b>{profile}</b> | gps=<b>{gps_src}</b> | ',
                f'mood=<b>{mood}</b> | mobility=<b>{mobility}</b>',
                '</p>',
                '<div class="links">',
                f'<a href="{base}export">📥 Export</a> | ',
                f'<a href="{base}metrics">📊 Metrics</a> | ',
                f'<a href="{base}zones">🗺️ Zones</a>',
                '</div>',
                self._ui_actions(),
                self._ui_status(),
                self._ui_current_params(),
                self._ui_ucb_summary(),
                self._ui_channels(),
                self._ui_top_aps(),
            ]
            if self._gps_available and self._gps_zones:
                parts.append(self._ui_gps_zones())
            parts.append('</body></html>')
            return self._html_response(''.join(parts))
        except Exception as e:
            logging.exception(f'[envtune] webhook: {e}')
            body = ('<!DOCTYPE html><html><body><h1>Error</h1>'
                    f'<pre>{html.escape(repr(e))}</pre></body></html>')
            return self._html_response(body, status=500)

    def _ui_css(self):
        return '''
body{font-family:"Courier New",monospace;background:#0d0d0d;color:#b0b0b0;
     margin:0;padding:18px;font-size:13px}
h1{color:#00ff88;letter-spacing:2px;margin:0 0 4px 0}
h2{color:#00ccff;border-bottom:1px solid #1a3a3a;padding-bottom:4px;
   margin-top:22px}
p.subtitle{color:#666;margin:0 0 10px 0}
div.links{margin-bottom:20px}
a{color:#00ccff;text-decoration:none}
a:hover{text-decoration:underline}
table{border-collapse:collapse;width:100%;margin-bottom:18px;
      table-layout:auto}
th{background:#0a1a2a;color:#00ff88;padding:5px 8px;
   border:1px solid #1a3a3a;text-align:left;font-size:0.88em;
   white-space:nowrap}
td{padding:3px 8px;border:1px solid #1a1a1a;font-size:0.87em;
   vertical-align:top;word-break:break-word}
tr:hover td{background:#111820}
.good{color:#00ff88;font-weight:bold}
.warn{color:#ffaa00}
.bad{color:#ff4444}
.na{color:#444}
small{font-size:0.78em;color:#666}
[title]{cursor:help;border-bottom:1px dotted #444}
.actbar{margin:6px 0 12px 0}
.actbtn{font-family:inherit;font-size:0.85em;padding:6px 12px;
        border:1px solid #1a3a3a;background:#101820;color:#00ccff;
        cursor:pointer;border-radius:3px}
.actbtn.good{color:#00ff88;border-color:#003322}
.actbtn.warn{color:#ffaa00;border-color:#332200}
.actbtn:hover{background:#16242c}
ul.actionlog{list-style:none;padding:0;margin:6px 0 12px 0;
             font-size:0.82em;color:#888}
ul.actionlog li{padding:2px 0;border-bottom:1px dotted #1a1a1a}
'''

    @staticmethod
    def _fmt(v, spec='.3f', na='N/A'):
        if v is None:
            return f'<span class="na">{na}</span>'
        try:
            return format(float(v), spec)
        except (ValueError, TypeError):
            return html.escape(str(v))

    def _ui_status(self):
        elapsed_h = max(0.01,
            (time.monotonic() - self.session_start_mono) / 3600.0)
        lt = int(time.time() - self.last_shake.get('time', time.time()))
        lt_s = f'{lt//60}m{lt%60:02d}s' if lt >= 60 else f'{lt}s'
        temp = self.ema.get('temperature') or 0
        temp_cls = 'bad' if temp >= self.cfg['temp_critical'] else (
            'warn' if temp >= self.cfg['temp_warn'] else 'good')

        ret = '<h2>📊 Status</h2><table>'
        rows = [
            ('Plugin version',     f'v{self.__version__}',
             'EnvTune release version'),
            ('CPU profile',        self._profile_name,
             'Performance profile (auto-detected or manual)'),
            ('Epochs observed',    self.epochs_seen,
             'Epochs since plugin started'),
            ('🆕 UNIQUE lifetime',
             f'<span class="good" style="font-size:1.2em">'
             f'{self._lifetime_new_count}</span>',
             'Distinct BSSIDs ever captured. THIS IS THE GOAL.'),
            ('Lifetime handshakes (incl. dups)',
             f'{self.lifetime_handshakes}',
             'Total HS events across all sessions, including duplicates'),
            ('Session duration',   f'{elapsed_h:.2f}h',
             'How long this run has been active'),
            ('Time since last HS', lt_s,
             'Wall-clock time since most recent capture'),
            ('Unique pwns (sess)', len(self._captured_aps),
             'Distinct APs handshaked this session'),
            ('Pre-captured BSSIDs', len(self._captured_bssids),
             'BSSIDs already on disk (deprioritized)'),
            ('Cracked (wpa-sec)',  len(self._cracked_bssids),
             'BSSIDs with known password from wpa-sec potfile'),
            ('Whitelisted',
             f'{len(self._whitelist_macs)} MAC + {len(self._whitelist_ssids)} SSID',
             'Networks excluded from tracking'),
            ('Known APs',          len(self._known_aps),
             'In-memory AP intelligence cache'),
            ('Active channels',    self._active_channels,
             'Channels with currently visible APs'),
            ('GPS source',         self._gps_source or 'none',
             'How GPS data is being read'),
            ('Current zone',       self._current_zone or 'n/a',
             'GPS-derived zone ID for context-specific learning'),
            ('Battery',            (f'{self._battery_level:.0f}%'
                                    if self._battery_level else 'n/a'),
             'PiSugar battery level'),
            ('EMA APs visible',    self._fmt(self.ema.get('aps'), '.1f'),
             'Smoothed AP count'),
            ('EMA HS rate',        self._fmt(self.ema.get('hs_rate')),
             'Handshakes per attack (smoothed)'),
            ('EMA HS/min',         self._fmt(self.ema.get('hs_per_min')),
             'Handshakes per minute (smoothed)'),
            ('Adaptive HPM target',
             self._fmt(self._adaptive_hpm_target()),
             '90th-percentile of recent unique-HS/min — reward target'),
            ('Reward trend',       self._fmt(self._reward_trend),
             'Direction of recent reward EMA'),
            ('Best custom reward', self._fmt(self.best_reward),
             'All-time best epoch reward'),
            ('Temperature',
             f'<span class="{temp_cls}">{self._fmt(temp, ".1f")}°C</span>',
             'CPU temperature EMA'),
            ('Thermal throttle',
             (f'<span class="bad">ACTIVE</span>'
              if self._thermal_throttle else
              f'<span class="good">off</span>'),
             'Whether attack aggression is reduced for thermal safety'),
            ('Exploration boost',  self._exploration_boost,
             'Epochs left of elevated UCB exploration'),
            ('Stagnation streak',  self._stagnation_count,
             'Consecutive epochs below rolling-median reward'),
            ('Blind recovery',     self._blind_recovery,
             'Epochs left of gradual blind-panic recovery'),
            ('Nexmon crash watch', self._crash_suspect,
             'Suspicion counter for radio firmware crash'),
        ]
        for label, val, tip in rows:
            ret += (f'<tr><td><span title="{html.escape(tip)}">{label}</span></td>'
                    f'<td>{val}</td></tr>')
        ret += '</table>'
        return ret

    def _ui_current_params(self):
        # Defensive: agent or its config may be missing during early boot
        # or if a fork relocates personality data.
        try:
            p = (self._agent._config or {}).get('personality', {}) or {}
        except Exception:
            p = {}
        ret = '<h2>🎛️ Current Personality Parameters</h2><table>'
        ret += ('<tr><th>Parameter</th><th>Current</th>'
                '<th>Bounds</th><th>Status</th></tr>')
        for param, (lo, hi) in self.BOUNDS.items():
            tuned    = param in self._active_params
            cls      = '' if tuned else 'na'
            status   = ('<span class="good">tuning</span>' if tuned
                        else '<span class="na">not in fork</span>')
            sync_tag = ' 🔄' if param in self.BETTERCAP_SYNC_MAP else ''
            cur_val  = p.get(param, '?') if isinstance(p, dict) else '?'
            ret += (f'<tr class="{cls}"><td>{html.escape(param)}{sync_tag}</td>'
                    f'<td><b>{html.escape(str(cur_val))}</b></td>'
                    f'<td>[{lo},{hi}]</td><td>{status}</td></tr>')
        ret += '<tr><td colspan=4><small>🔄 = synced to bettercap '
        ret += 'in realtime via "set wifi.* N"</small></td></tr>'
        ret += '</table>'
        return ret

    def _ui_ucb_summary(self):
        aps_ema = self.ema.get('aps') or 0
        state   = self._compute_state(aps_ema)
        ret  = (f'<h2>🧠 UCB Learning — current state: '
                f'<b style="color:#ff0">{html.escape(str(state))}</b></h2><table>')
        ret += ('<tr><th>Param</th><th>Best arm</th>'
                '<th>Mean rwd</th><th>Window n</th>'
                '<th>All arms (n:mean)</th></tr>')
        # Snapshot the per-state UCB tables under lock so concurrent updates
        # don't mutate dicts mid-iteration.
        with self._state_lock:
            for param in list(self.UCB_ARMS.keys()):
                if param in self._active_params:
                    self._ensure_state(param, state)
            snap = {}
            for param, arms in self.UCB_ARMS.items():
                if param not in self._active_params:
                    continue
                tbl_state = self.ucb_table.get(param, {}).get(state, {})
                snap[param] = {
                    arm: list(tbl_state.get(arm, {}).get('rewards', []))
                    for arm in arms
                }
        for param, arms in self.UCB_ARMS.items():
            if param not in snap:
                continue
            arm_snap  = snap[param]
            best_arm  = None
            best_mean = -1.0
            best_wn   = 0
            parts     = []
            for arm in arms:
                rewards = arm_snap.get(arm, [])
                wn      = len(rewards)
                mean    = (sum(rewards) / wn) if wn > 0 else 0.0
                parts.append(f'{arm}({wn}:{mean:.2f})')
                if wn > 0 and mean > best_mean:
                    best_mean, best_arm, best_wn = mean, arm, wn
            if best_arm is not None:
                ret += (f'<tr><td>{html.escape(param)}</td>'
                        f'<td class="good"><b>{html.escape(str(best_arm))}</b></td>'
                        f'<td>{best_mean:.3f}</td><td>{best_wn}</td>'
                        f'<td><small>{html.escape(" ".join(parts))}</small></td></tr>')
            else:
                ret += (f'<tr><td>{html.escape(param)}</td>'
                        f'<td colspan=3 class="na">exploring…</td>'
                        f'<td><small>{html.escape(" ".join(parts))}</small></td></tr>')
        ret += '</table>'
        return ret

    def _ui_channels(self):
        ret  = '<h2>📡 Channel Productivity (Lifetime)</h2><table>'
        ret += ('<tr><th>Ch</th><th>HS</th><th>Passive HS</th>'
                '<th>Cracked</th><th>Assocs</th><th>Deauths</th>'
                '<th>Clients</th><th>Visits</th><th>Wasted</th>'
                '<th>Free</th><th>Dead⚡</th><th>Score</th></tr>')
        # FIX: snapshot under lock — UI reads concurrently with event handlers.
        with self._state_lock:
            ch_lt_snap = {c: dict(v) for c, v in self._ch_lt.items()}
        chs = sorted(ch_lt_snap.keys(),
                     key=lambda c: -ch_lt_snap[c]['hs'])[:25]
        for ch in chs:
            d   = ch_lt_snap[ch]
            sc  = self._ch_score(ch)
            nol = '🔵' if ch in self.NON_OVERLAPPING else ''
            fr  = '✨' if ch in self._free_channels else ''
            ret += (f'<tr><td>{ch}{nol}{fr}</td>'
                    f'<td class="good"><b>{d["hs"]}</b></td>'
                    f'<td>{d.get("passive_hs", 0)}</td>'
                    f'<td>{d.get("cracked", 0)}</td>'
                    f'<td>{d["assocs"]}</td>'
                    f'<td>{d["deauths"]}</td>'
                    f'<td>{d["clients"]}</td>'
                    f'<td>{d["visits"]}</td>'
                    f'<td class="{"bad" if d["wasted"] > 10 else "warn"}">'
                    f'{d["wasted"]}</td>'
                    f'<td>{d.get("free_seen", 0)}</td>'
                    f'<td class="bad">{self._dead_lt.get(ch, 0)}</td>'
                    f'<td>{sc:.2f}</td></tr>')
        ret += ('<tr><td colspan=12><small>'
                '🔵 = non-overlapping channel · '
                '✨ = recently reported free</small></td></tr>')
        ret += '</table>'
        return ret

    def _ui_top_aps(self):
        ret  = '<h2>🎯 AP Intelligence (session)</h2><table>'
        ret += ('<tr><th>SSID</th><th>BSSID</th><th>Ch</th>'
                '<th>RSSI</th><th>Trend</th><th>Clients</th>'
                '<th>HS</th><th>Attacks</th><th>Eff.</th>'
                '<th>Cooldown</th><th>Flags</th></tr>')
        # Snapshot under lock so we don't iterate a dict another thread is
        # mutating (handshake handler / on_wifi_update). Shallow-copy each
        # AP record because helpers below access its fields after release.
        with self._state_lock:
            ap_snap = [(k, dict(v)) for k, v in self._known_aps.items()]
            ep      = self.epochs_seen
        sorted_aps = sorted(
            ap_snap,
            key=lambda x: (-x[1].get('AT_handshake', 0),
                           -self._ap_priority_score(x[0]))
        )[:50]
        for apID, ap in sorted_aps:
            eff     = ap.get('AT_efficiency', 0.0)
            eff_cls = ('good' if eff >= 0.1 else
                      ('warn' if eff > 0 else 'bad'))
            trend   = self._rssi_trend(apID)
            t_str   = (f'<span class="good">▲{trend:+.1f}</span>' if trend > 1
                       else (f'<span class="bad">▼{trend:+.1f}</span>'
                             if trend < -1 else '—'))
            cd_left = max(0, ap.get('AT_cooldown_until', 0) - ep)
            ncl     = ap.get('AT_clients', 0)
            flags   = []
            if ap.get('AT_pmf_detected'):     flags.append('PMF')
            if ap.get('AT_already_captured'): flags.append('✓Cap')
            if ap.get('AT_cracked'):          flags.append('🔓')
            host    = html.escape(str(ap.get('hostname', '?'))[:24])
            mac     = html.escape(str(ap.get('mac', '?')))
            chan    = html.escape(str(ap.get('channel', '?')))
            rssi    = html.escape(str(ap.get('rssi', '?')))
            ret += (f'<tr>'
                    f'<td>{host}</td>'
                    f'<td><small>{mac}</small></td>'
                    f'<td>{chan}</td>'
                    f'<td>{rssi}</td>'
                    f'<td>{t_str}</td>'
                    f'<td>{"🧑" * min(ncl, 5)}{ncl}</td>'
                    f'<td class="good"><b>{ap.get("AT_handshake", 0)}</b></td>'
                    f'<td>{ap.get("AT_attacks", 0)}</td>'
                    f'<td class="{eff_cls}">{eff:.2f}</td>'
                    f'<td>{"⏸ " + str(cd_left) + "ep" if cd_left > 0 else ""}</td>'
                    f'<td>{html.escape(" ".join(flags))}</td>'
                    f'</tr>')
        ret += '</table>'
        return ret

    def _ui_gps_zones(self):
        ret  = '<h2>🗺️ GPS Zone Productivity</h2><table>'
        ret += ('<tr><th>Zone</th><th>HS</th><th>Attacks</th>'
                '<th>Visits</th><th>Top channels</th><th>Last seen</th></tr>')
        # Deep-snapshot zones under lock so per-zone channel dicts are stable.
        with self._state_lock:
            zones_snap = [
                (zk, {
                    'hs': z.get('hs', 0),
                    'attacks': z.get('attacks', 0),
                    'visits': z.get('visits', 0),
                    'last_seen': z.get('last_seen', 0),
                    'channels': dict(z.get('channels', {})),
                })
                for zk, z in self._gps_zones.items()
            ]
        zones = sorted(zones_snap, key=lambda kv: -kv[1]['hs'])[:30]
        now = time.time()
        for zk, zd in zones:
            top = sorted(zd['channels'].items(),
                         key=lambda x: -x[1])[:3]
            top_s = ', '.join(f'{c}:{n}' for c, n in top) or '—'
            ago = ''
            if zd.get('last_seen', 0):
                secs = int(now - zd['last_seen'])
                ago = (f'{secs//3600}h{(secs%3600)//60}m ago'
                       if secs > 3600 else f'{secs//60}m ago')
            ret += (f'<tr><td><small>{html.escape(zk)}</small></td>'
                    f'<td class="good"><b>{zd["hs"]}</b></td>'
                    f'<td>{zd["attacks"]}</td>'
                    f'<td>{zd["visits"]}</td>'
                    f'<td>{html.escape(top_s)}</td>'
                    f'<td><small>{html.escape(ago)}</small></td></tr>')
        ret += '</table>'
        return ret

    # ── Endpoints ─────────────────────────────────────────────────────────

    def _endpoint_export(self):
        """Full state JSON for backup or sharing as community prior."""
        try:
            data = self._build_state_snapshot()
            resp = make_response(json.dumps(data, indent=2, default=str), 200)
            resp.headers['Content-Type'] = 'application/json; charset=utf-8'
            resp.headers['Cache-Control'] = 'no-store'
            return resp
        except Exception as e:
            resp = make_response(f'Error: {html.escape(str(e))}', 500)
            resp.headers['Content-Type'] = 'text/plain; charset=utf-8'
            return resp

    def _endpoint_metrics(self):
        """Prometheus-compatible metrics — snapshot under lock, no iteration."""
        try:
            with self._state_lock:
                lifetime_hs   = self.lifetime_handshakes
                lifetime_uniq = self._lifetime_new_count
                sess_uniq     = len(self._captured_aps)
                sess_dups     = max(0, self.session_handshakes - sess_uniq)
                pre_cap       = len(self._captured_bssids)
                cracked       = len(self._cracked_bssids)
                known_aps     = len(self._known_aps)
                gps_zones     = len(self._gps_zones)
                free_ch       = len(self._free_channels)
                active_ch     = self._active_channels
                stagnation    = self._stagnation_count
                blind_rec     = self._blind_recovery
                explor_boost  = self._exploration_boost
                crash_susp    = self._crash_suspect
                thermal       = 1 if self._thermal_throttle else 0
                temp_ema      = self.ema.get('temperature') or 0
                hpm           = self.ema.get('hs_per_min') or 0
                aps_ema       = self.ema.get('aps') or 0
                hs_rate       = self.ema.get('hs_rate') or 0
                target_hpm    = self._adaptive_hpm_target() or 0
                trend         = self._reward_trend or 0
                best_rwd      = self.best_reward
                epochs_seen   = self.epochs_seen
                session_mono  = self.session_start_mono
                whitelist     = len(self._whitelist_macs) + len(self._whitelist_ssids)
                save_q        = self._save_queue.qsize() if hasattr(self, '_save_queue') else 0
            uptime_s = max(0.0, time.monotonic() - session_mono)
            lines = [
                '# HELP envtune_lifetime_handshakes Total HS captured ever (incl dups)',
                '# TYPE envtune_lifetime_handshakes counter',
                f'envtune_lifetime_handshakes {lifetime_hs}',
                '# HELP envtune_unique_lifetime_bssids Distinct BSSIDs ever captured (THE GOAL)',
                '# TYPE envtune_unique_lifetime_bssids counter',
                f'envtune_unique_lifetime_bssids {lifetime_uniq}',
                '# HELP envtune_session_unique Distinct BSSIDs captured this session',
                '# TYPE envtune_session_unique gauge',
                f'envtune_session_unique {sess_uniq}',
                '# HELP envtune_session_duplicates Duplicate handshakes this session',
                '# TYPE envtune_session_duplicates gauge',
                f'envtune_session_duplicates {sess_dups}',
                '# HELP envtune_precaptured_bssids Pre-captured BSSIDs from .pcap files',
                '# TYPE envtune_precaptured_bssids gauge',
                f'envtune_precaptured_bssids {pre_cap}',
                '# HELP envtune_cracked_bssids BSSIDs known cracked via wpa-sec potfile',
                '# TYPE envtune_cracked_bssids gauge',
                f'envtune_cracked_bssids {cracked}',
                '# HELP envtune_whitelisted Networks excluded from tracking',
                '# TYPE envtune_whitelisted gauge',
                f'envtune_whitelisted {whitelist}',
                '# HELP envtune_known_aps APs tracked in memory',
                '# TYPE envtune_known_aps gauge',
                f'envtune_known_aps {known_aps}',
                '# HELP envtune_active_channels Channels with currently visible APs',
                '# TYPE envtune_active_channels gauge',
                f'envtune_active_channels {active_ch}',
                '# HELP envtune_free_channels Channels recently reported as free',
                '# TYPE envtune_free_channels gauge',
                f'envtune_free_channels {free_ch}',
                '# HELP envtune_temperature_celsius CPU temperature EMA',
                '# TYPE envtune_temperature_celsius gauge',
                f'envtune_temperature_celsius {temp_ema}',
                '# HELP envtune_hs_per_min Smoothed handshakes per minute',
                '# TYPE envtune_hs_per_min gauge',
                f'envtune_hs_per_min {hpm}',
                '# HELP envtune_target_hpm Adaptive HPM target (90th percentile)',
                '# TYPE envtune_target_hpm gauge',
                f'envtune_target_hpm {target_hpm}',
                '# HELP envtune_aps_visible_ema Smoothed visible-AP count',
                '# TYPE envtune_aps_visible_ema gauge',
                f'envtune_aps_visible_ema {aps_ema}',
                '# HELP envtune_hs_per_attack Smoothed handshakes per attack',
                '# TYPE envtune_hs_per_attack gauge',
                f'envtune_hs_per_attack {hs_rate}',
                '# HELP envtune_reward_trend Direction of recent reward EMA',
                '# TYPE envtune_reward_trend gauge',
                f'envtune_reward_trend {trend}',
                '# HELP envtune_best_reward All-time best epoch reward',
                '# TYPE envtune_best_reward gauge',
                f'envtune_best_reward {best_rwd}',
                '# HELP envtune_epochs_seen Epochs since plugin started',
                '# TYPE envtune_epochs_seen counter',
                f'envtune_epochs_seen {epochs_seen}',
                '# HELP envtune_thermal_throttle 1 if thermal throttle active',
                '# TYPE envtune_thermal_throttle gauge',
                f'envtune_thermal_throttle {thermal}',
                '# HELP envtune_stagnation_streak Consecutive sub-median epochs',
                '# TYPE envtune_stagnation_streak gauge',
                f'envtune_stagnation_streak {stagnation}',
                '# HELP envtune_blind_recovery_left Epochs left of blind-panic recovery',
                '# TYPE envtune_blind_recovery_left gauge',
                f'envtune_blind_recovery_left {blind_rec}',
                '# HELP envtune_exploration_boost_left Epochs left of elevated UCB exploration',
                '# TYPE envtune_exploration_boost_left gauge',
                f'envtune_exploration_boost_left {explor_boost}',
                '# HELP envtune_crash_suspect Suspected nexmon crash counter',
                '# TYPE envtune_crash_suspect gauge',
                f'envtune_crash_suspect {crash_susp}',
                '# HELP envtune_gps_zones Distinct GPS zones learned',
                '# TYPE envtune_gps_zones gauge',
                f'envtune_gps_zones {gps_zones}',
                '# HELP envtune_save_queue_depth Pending state-save tasks',
                '# TYPE envtune_save_queue_depth gauge',
                f'envtune_save_queue_depth {save_q}',
                '# HELP envtune_uptime_seconds Session uptime',
                '# TYPE envtune_uptime_seconds counter',
                f'envtune_uptime_seconds {uptime_s:.1f}',
            ]
            resp = make_response('\n'.join(lines) + '\n', 200)
            resp.headers['Content-Type'] = 'text/plain; version=0.0.4; charset=utf-8'
            resp.headers['Cache-Control'] = 'no-store'
            return resp
        except Exception as e:
            resp = make_response(f'Error: {html.escape(str(e))}', 500)
            resp.headers['Content-Type'] = 'text/plain; charset=utf-8'
            return resp

    def _endpoint_zones(self):
        """GPS zones JSON for external mapping tools."""
        try:
            with self._state_lock:
                data = {
                    zk: {
                        'hs': z.get('hs', 0),
                        'attacks': z.get('attacks', 0),
                        'visits': z.get('visits', 0),
                        'last_seen': z.get('last_seen', 0),
                        'channels': dict(z.get('channels', {})),
                    }
                    for zk, z in self._gps_zones.items()
                }
            resp = make_response(json.dumps(data, indent=2, default=str), 200)
            resp.headers['Content-Type'] = 'application/json; charset=utf-8'
            resp.headers['Cache-Control'] = 'no-store'
            return resp
        except Exception as e:
            resp = make_response(f'Error: {html.escape(str(e))}', 500)
            resp.headers['Content-Type'] = 'text/plain; charset=utf-8'
            return resp

    # ── Actions panel & POST handlers ─────────────────────────────────────

    def _ui_actions(self):
        """HTML form panel for operator-driven actions. Each form posts a
        CSRF token bound to the running process."""
        token = html.escape(self._csrf_token)
        # Render last 8 actions (newest first)
        with self._state_lock:
            log_items = list(self._action_log)[-8:][::-1]
        log_html = ''
        if log_items:
            log_html = '<ul class="actionlog">'
            for ts, name, ok, msg in log_items:
                cls = 'good' if ok else 'bad'
                t   = time.strftime('%H:%M:%S', time.localtime(ts))
                log_html += (f'<li><span class="{cls}">{t}</span> '
                             f'<b>{html.escape(name)}</b> — '
                             f'{html.escape(msg)}</li>')
            log_html += '</ul>'

        # Use the plugin's absolute mount point so the form action does
        # NOT resolve against `/plugins/` if the user reached the page
        # without a trailing slash (which would hit `/plugins/force-save`
        # and 404).
        base = html.escape(self._plugin_base())

        def _form(action, label, hint, cls='warn'):
            return (
                f'<form method="POST" action="{base}{html.escape(action)}" '
                f'style="display:inline-block;margin:2px 6px 2px 0">'
                f'<input type="hidden" name="csrf" value="{token}">'
                f'<button type="submit" class="actbtn {cls}" '
                f'title="{html.escape(hint)}">{html.escape(label)}</button>'
                f'</form>'
            )
        ret = '<h2>🛠 Actions</h2>'
        ret += '<div class="actbar">'
        ret += _form('force-save',
                     '💾 Force save',
                     'Flush plugin state JSON to disk now', 'good')
        ret += _form('rescan-potfile',
                     '🔓 Rescan wpa-sec',
                     'Re-read /root/handshakes/wpa-sec.cracked.potfile',
                     'good')
        ret += _form('reset-stagnation',
                     '🔄 Reset stagnation',
                     'Clear stagnation streak & decision buffer; re-explore',
                     'warn')
        ret += _form('reload-whitelist',
                     '⛔ Reload whitelist',
                     'Reload main_whitelist & main_handshakes from config',
                     'warn')
        ret += _form('clear-blind',
                     '👁 Clear blind-panic',
                     'Drop blind-recovery counter to zero', 'warn')
        ret += '</div>'
        ret += log_html
        return ret

    def _record_action(self, name, ok, msg):
        with self._state_lock:
            self._action_log.append((time.time(), name, bool(ok), str(msg)))

    def _verify_csrf(self, request):
        try:
            tok = ''
            if hasattr(request, 'form'):
                tok = request.form.get('csrf', '') or ''
            if not tok and hasattr(request, 'values'):
                tok = request.values.get('csrf', '') or ''
            if not tok and hasattr(request, 'headers'):
                tok = request.headers.get('X-CSRF-Token', '') or ''
            return bool(tok) and hmac.compare_digest(tok, self._csrf_token)
        except Exception:
            return False

    def _post_redirect(self, action, ok, msg, status=303):
        # Always redirect back to dashboard so the form-submission browser
        # context stays clean (no page-reload re-POST). Action result is
        # visible in the action log.
        self._record_action(action, ok, msg)
        # Absolute path — `./` would resolve wrong if browser landed on
        # /plugins/envtune (no trailing slash) before the POST.
        base = self._plugin_base()
        body = (f'<!DOCTYPE html><html><head>'
                f'<meta http-equiv="refresh" content="1; url={html.escape(base)}">'
                f'</head><body><p>{html.escape(msg)}</p>'
                f'<p><a href="{html.escape(base)}">→ back</a></p></body></html>')
        resp = make_response(body, status)
        resp.headers['Content-Type'] = 'text/html; charset=utf-8'
        resp.headers['Location'] = base
        resp.headers['Cache-Control'] = 'no-store'
        return resp

    def _handle_post(self, path, request):
        if not self._verify_csrf(request):
            return self._html_response(
                '<!DOCTYPE html><html><body><h1>403</h1>'
                '<p>CSRF token invalid or missing.</p></body></html>',
                status=403)
        try:
            if path == 'force-save':
                self._enqueue_save(reason='manual')
                return self._post_redirect(
                    'force-save', True,
                    'State save enqueued.')
            if path == 'rescan-potfile':
                cracked = self._scan_cracked_potfile()
                with self._state_lock:
                    added = len(cracked - self._cracked_bssids)
                    self._cracked_bssids = cracked
                    # Mark already-known APs as cracked so the targeting loop
                    # picks the change up immediately.
                    for k, ap in self._known_aps.items():
                        if self._mac_norm(ap.get('mac', '')) in cracked:
                            ap['AT_cracked'] = True
                self._enqueue_save(reason='potfile-rescan')
                return self._post_redirect(
                    'rescan-potfile', True,
                    f'Potfile rescanned — {len(cracked)} cracked '
                    f'BSSIDs ({added} new).')
            if path == 'reset-stagnation':
                with self._state_lock:
                    self._stagnation_count = 0
                    self._exploration_boost = max(self._exploration_boost,
                                                  self.cfg.get(
                                                      'stagnation_boost_epochs',
                                                      30))
                    if hasattr(self, '_decision_buffer'):
                        try:
                            self._decision_buffer.clear()
                        except Exception:
                            pass
                return self._post_redirect(
                    'reset-stagnation', True,
                    'Stagnation streak reset and exploration boosted.')
            if path == 'reload-whitelist':
                if self._agent is not None:
                    self._load_whitelist(self._agent)
                with self._state_lock:
                    n_mac  = len(self._whitelist_macs)
                    n_ssid = len(self._whitelist_ssids)
                return self._post_redirect(
                    'reload-whitelist', True,
                    f'Whitelist reloaded ({n_mac} MAC, {n_ssid} SSID).')
            if path == 'clear-blind':
                with self._state_lock:
                    prior = self._blind_recovery
                    self._blind_recovery = 0
                    self._crash_suspect = 0
                return self._post_redirect(
                    'clear-blind', True,
                    f'Blind-panic cleared (was {prior}).')
        except Exception as e:
            logging.exception(f'[envtune] POST {path}: {e}')
            return self._post_redirect(path, False, repr(e), status=500)
        return self._html_response(
            '<!DOCTYPE html><html><body><h1>404</h1>'
            f'<p>Unknown action: {html.escape(str(path))}</p>'
            '</body></html>',
            status=404)
