This is a old repo, you can find newer updates on: https://github.com/adi170-alt/pwnagotchi_plugins
# EnvTune

**Adaptive Environment Tuner for Pwnagotchi — drop-in replacement for the removed AI.**

Maximises *unique* (lifetime-new) BSSID handshake captures by learning optimal `personality` parameters for each environment using **Sliding-Window UCB1 with empirical-Bayes shrinkage**. Built specifically for the [jayofelony/pwnagotchi](https://github.com/jayofelony/pwnagotchi) `noai` branch.

- ≈ 2-3 % CPU on a Pi Zero 2 W
- No neural net → cannot crash the WiFi firmware
- Stdlib only (no extra `pip` packages)
- State persists across reboots and plugin upgrades

---

## Why this exists

Jay removed the A2C neural network because it destabilised the WiFi firmware and drained batteries. The stock `noai` build is solid but runs on **fixed parameters** — it cannot adapt to your routes, times, or environments. EnvTune fills that gap with lightweight, predictable ML that **only ever tunes parameters pwnagotchi already supports**.

The optimisation target is unambiguous: **the count of BSSIDs you have NEVER captured before, ever, across all sessions.** Catching the same network ten times yields the same reward as catching it once. Brand-new captures earn full reward.

---

## How it works (in 60 seconds)

1. Each epoch, EnvTune classifies the current environment into one of **108 contexts** (AP density × time-of-day × reward trend × mobility).
2. For each of **14 personality parameters**, a Sliding-Window UCB1 bandit picks a value ("arm") for the current context. **Annealed empirical-Bayes shrinkage** pulls every cell toward its parent group (states sharing 2+ context dims) with weight `n / (n+k)`, where `k` itself decays from 5 → 1 over the first 500 real samples — so cold contexts inherit useful priors *and* genuinely-better arms aren't permanently anchored to a mediocre prior late in the session.
3. Reward is computed `reward_delay` epochs later (default 3, **adaptive**: −1 in dense areas, +1 in sparse).
4. The reward signal uses Hill-style saturation `r = ratio / (ratio + 1)` so a target-hit clearly outranks the cold-start prior — UCB can actually distinguish "did nothing" from "did well" from "knocked it out of the park".
5. Channels are scheduled by combined lifetime productivity + live uncaptured-AP opportunity + a **per-channel efficiency multiplier** (low-yield channels are deprioritised even if they have many APs). Already-captured BSSIDs are pushed to bettercap's `wifi.assoc.skip` / `wifi.deauth.skip` so no airtime is wasted on duplicates.

After ~150–250 epochs in a given environment the plugin starts consistently picking the right parameters. The sliding window means stale memory never overrides fresh evidence.

---

## Key capabilities

### Learning
- **14-parameter SW-UCB1** with annealed exploration constant
- **Annealed empirical-Bayes shrinkage** — every UCB cell blends local mean with parent-group mean, with shrinkage strength itself decaying as the table matures (heavy at cold-start, light once enough real data exists). Telemetry showed v1.1's fixed `k=5` was permanently dragging high-mean low-n arms back toward the prior; v1.2 lets them break free as evidence accumulates.
- **Hierarchical priors** so rare contexts benefit from common ones
- **Time-of-day priors** seed reasonable defaults instantly on first install
- **Saturation-aware exploration boost** — when >80 % of visible non-cracked APs are already captured, exploration is widened so the bandit re-tests in a different direction

### Targeting
- **Fresh-session BSSID priority** — a brand-new BSSID gets a +1.5 priority bonus that decays linearly over 8 epochs
- **Client-aware deauth windows** — clients seen ≤1 epoch ago stack an extra +2.0 per client
- **Per-AP cooldown** on persistent non-responders + tier-based AP eviction (cracked-captured first, fresh-untried last)
- **PMF detection** with 200-epoch re-evaluation
- **Already-captured detection** from `/root/handshakes/` + persisted state, with **immediate** skip-list push on handshake (not next epoch)
- **wpa-sec cracked feedback** — skips already-cracked networks; periodic potfile rescan every 100 epochs
- **Whitelist respect** — whitelisted APs never enter UCB statistics
- **Handshake-dir watchdog** — every 200 epochs the handshake directory is rescanned to catch externally-added pcaps

### Channels
- **Lifetime + live opportunity scoring** — historical productivity weighted with currently-visible uncaptured APs
- **Per-channel efficiency multiplier** (0.5×–1.5×) — telemetry showed low-yield channels were over-visited; this rebalances toward channels that actually convert
- **5 GHz-aware recon timing** — when >30 % of visible APs are 5 GHz, recon shrinks to compensate for slower band scans
- **Free-channel opportunism** via `on_free_channel` callback
- **Dead-channel cooldown** with automatic recovery

### Spatial / temporal
- **GPS zone-aware learning** — auto-detects TheyLive or stock `gps`, no config required
- **Zone-keyed channel histogram** (HS-keyed, not visit-keyed) — re-entering a known zone immediately favours the channels that produced handshakes there
- **Stationary vs mobile detection** via speed and AP turnover; UCB priors adjust automatically
- **GPS zone LRU cap (500)** with tier-based eviction — zones with ≥50 attacks and zero handshakes are evicted *before* never-touched zones, since those untouched zones may still produce later
- **Heatmap of captures** with per-zone productivity scoring

### Safety
- **Thermal safety** — backs off at 70 °C, hard-throttles at 78 °C
- **Nexmon crash detection** + automatic backoff (won't fight a wedged firmware)
- **Blind-panic state machine** — radio sees nothing → forces aggressive scan params, throttles preserved (no double-throttle erasure)
- **EMA input clamps** — every smoothed metric has sane bounds, so a single rogue native-reward sample can't poison the EMA forever (we observed `reward = -8.5e15` in real telemetry; this safeguard prevents a recurrence)

### Operations
- **Async state save** — no SD-card stalls mid-epoch
- **Atomic writes** via `tempfile` + `os.replace` + `fsync`
- **Versioned + migrated state** — survives plugin upgrades, schema bumps cleanly
- **Five CPU profiles** — `minimal`, `light`, `balanced`, `aggressive`, `beast`
- **PiSugar awareness** (optional, graceful)

### Web UI
- **Modern dashboard** at `/plugins/envtune/`
- **Five operator actions** (force-save, rescan-potfile, reset-stagnation, reload-whitelist, clear-blind) — each protected with a per-process CSRF token, all relative URLs anchored via `<base href="…">` so they work whether you visited the page with or without a trailing slash
- **25 Prometheus counters** for Grafana dashboards
- **All dynamic values HTML-escaped** — attacker-controlled SSIDs cannot inject markup or trigger Jinja evaluation

---

## Requirements

| | |
|---|---|
| Pwnagotchi | jayofelony/pwnagotchi (noai branch) |
| Python     | 3.7+ (already in the stock image) |
| Extra deps | none |

---

## Installation

1. Copy `envtune.py` to:
   ```
   /usr/local/share/pwnagotchi/custom-plugins/envtune.py
   ```

2. Add to `/etc/pwnagotchi/config.toml`:
   ```toml
   main.plugins.envtune.enabled = true
   ```

3. *(Optional)* Pick a CPU profile for your hardware:
   ```toml
   main.plugins.envtune.cpu_profile = "balanced"
   # choices: minimal | light | balanced | aggressive | beast
   # default: auto-detected from /proc/cpuinfo
   ```

4. *(Optional)* Turn off stock `auto_tune` if you used it — EnvTune replaces it.

5. Reboot. First **5 epochs = warmup** (observation only). After ~**150–250 epochs** you should see consistent gains in unique captures per session.

GPS and PiSugar are auto-detected — no extra config.

---

## CPU profiles

| Profile     | UCB window | Zone res | AP track cap | Extra ch | Proactive | Recommended HW |
|-------------|------------|----------|--------------|----------|-----------|----------------|
| minimal     | 20         | 300 m    | 150          | 2        | off       | very weak / battery saver |
| light       | 30         | 200 m    | 250          | 3        | off       | Pi Zero |
| balanced    | 40         | 150 m    | 400          | 3        | on        | Pi Zero 2 W, Pi 3 |
| aggressive  | 60         | 100 m    | 600          | 4        | on        | Pi 4 |
| beast       | 80         | 75 m     | 1000         | 5        | on        | Pi 5 |

---

## Config recipes

**Aggressive wardriving:**
```toml
main.plugins.envtune.cpu_profile     = "aggressive"
main.plugins.envtune.temp_critical   = 80.0
main.plugins.envtune.extra_channels  = 5
```

**Stealthy home use:**
```toml
main.plugins.envtune.cpu_profile             = "light"
main.plugins.envtune.ucb_c                   = 1.0
main.plugins.envtune.opportunistic_overrides = false
```

**Max learning on strong hardware:**
```toml
main.plugins.envtune.cpu_profile         = "beast"
main.plugins.envtune.ucb_window          = 80
main.plugins.envtune.save_every_n_epochs = 10
```

**Override the annealed shrinkage with a fixed value (legacy v1.1 behaviour):**
```toml
main.plugins.envtune.ucb_shrinkage_k = 2.0   # default: annealed 5.0 → 1.0
```

**Tune the anneal curve directly:**
```toml
main.plugins.envtune.ucb_shrinkage_k_max         = 5.0  # cold-start strength
main.plugins.envtune.ucb_shrinkage_k_min         = 1.0  # late-game floor
main.plugins.envtune.ucb_shrinkage_anneal_samples = 500 # samples to fully anneal
```

**Frequent disk rescans (busy environments):**
```toml
main.plugins.envtune.potfile_rescan_every_n   = 50    # default 100
main.plugins.envtune.handshake_rescan_every_n = 100   # default 200
```

All `DEFAULTS` keys can be overridden under `main.plugins.envtune.<key>`.

---

## Tuned parameters (UCB arms)

| Parameter                     | Arms                          |
|-------------------------------|-------------------------------|
| `min_rssi`                    | -85, -80, -75, -70, -65       |
| `hop_recon_time`              | 4, 6, 8, 10, 12, 15           |
| `min_recon_time`              | 2, 3, 5, 7, 10                |
| `recon_time`                  | 15, 20, 25, 30, 35, 45        |
| `max_interactions`            | 2, 3, 4, 5, 6                 |
| `ap_ttl`                      | 60, 120, 180, 300, 600        |
| `sta_ttl`                     | 120, 300, 600, 900            |
| `max_misses_for_recon`        | 3, 5, 7, 10                   |
| `max_inactive_scale`          | 2, 3, 5                       |
| `recon_inactive_multiplier`   | 1, 2, 3                       |
| `throttle_a`                  | 0.2, 0.4, 0.6, 0.8, 1.0       |
| `throttle_d`                  | 0.3, 0.5, 0.7, 0.9, 1.2       |
| `bored_num_epochs`            | 10, 15, 20, 25                |
| `sad_num_epochs`              | 15, 20, 25, 30                |

Hard `BOUNDS` clamp every value during panic / thermal modes — UCB never exits the safe envelope.

---

## Reward function

```
r =  0.60 · lifetime-new HS / min     (primary objective, Hill-saturated)
   + 0.10 · new APs discovered        (exploration value)
   + 0.08 · unique-per-attack         (duplicates don't help)
   + 0.06 · 1 - missed_rate           (efficiency)
   + 0.05 · active_ratio              (we're working)
   + 0.04 · hop diversity             (coverage)
   + 0.03 · native pwnagotchi reward  (loose alignment)
   + 0.04 · "underlying work" proxy   (avoids 0-HS deadzones)
   - 0.05 · inactive_ratio            (penalty for stalls)
   - 0.07 · blind_ratio               (penalty for radio sees nothing)
   - 0.04 · sad_for_epochs / total    (mood penalty, gated ≥5 epochs)
   - 0.03 · bored_for_epochs / total  (mood penalty, gated ≥5 epochs)
```

Plus an **activity floor of 0.01** when there were any interactions but zero new captures, so UCB still distinguishes "tried something but failed" from "did nothing at all".

The primary term uses Hill saturation:

```
new_term = ratio / (ratio + 1)
```

with `ratio = (lifetime-new captures / min) / target`. Target is the **90th percentile** of recent unique-HS-per-min — only the very best epochs raise the bar. This gives:

| ratio       | new_term | weighted contribution |
|-------------|----------|-----------------------|
| 0.00        | 0.000    | 0.000                 |
| 0.50        | 0.333    | 0.200                 |
| 1.00 (target) | 0.500  | 0.300                 |
| 2.00        | 0.667    | 0.400                 |
| 4.00        | 0.800    | 0.480                 |
| 8.00        | 0.889    | 0.533                 |

A target-hit weighted contribution of 0.30 sits cleanly above the 0.30 cold-start UCB prior, so the bandit can actually rank arms.

Reward components are stashed in a per-epoch breakdown dict for debug logging.

---

## Web UI

```
http://<pwnagotchi-ip>:8080/plugins/envtune/
```

Live stats, the full UCB learning table, channel productivity, AP intelligence, GPS zones, thermal/battery status. Hover any value for an explanation. Five buttons run operator actions; each requires a CSRF token bound to the running plugin process.

### Endpoints

| Path                       | Method | Returns                      |
|----------------------------|--------|------------------------------|
| `/plugins/envtune/`        | GET    | Dashboard (HTML)             |
| `/plugins/envtune/export`  | GET    | Full state (JSON)            |
| `/plugins/envtune/metrics` | GET    | Prometheus-style metrics (25 counters) |
| `/plugins/envtune/zones`   | GET    | Per-zone productivity (JSON) |

### Operator actions (POST, CSRF-protected)

| Action               | Purpose                                                  |
|----------------------|----------------------------------------------------------|
| `force-save`         | Flush plugin state JSON to disk now                      |
| `rescan-potfile`     | Re-read `/root/handshakes/wpa-sec.cracked.potfile`       |
| `reset-stagnation`   | Clear stagnation streak & decision buffer; re-explore    |
| `reload-whitelist`   | Reload `main.whitelist` and handshake list from config   |
| `clear-blind`        | Drop blind-recovery counter to zero                      |

The dashboard emits an absolute mount-point in the `<base href>` and on every form action so the buttons keep working whether you reach the dashboard with or without a trailing slash.

---

## State & persistence

- State file: `/etc/pwnagotchi/envtune_state.json`
- Atomic writes via `tempfile` + `os.replace` + `fsync`
- Async save thread coalesces rapid requests — no SD-card stall mid-epoch
- Schema-versioned (`STATE_SCHEMA_VERSION`); migrates older saves on load
- **EMA values are sanitised on load** — out-of-range values from older versions (or one-off bad samples) are dropped and re-seeded on the next epoch
- Persisted: EMAs, lifetime totals, captured + cracked BSSID sets, channel lifetime stats, dead-channel ledger, GPS zones, UCB Q-tables, best-known reward + settings

---

## Comparison with predecessors

| | stock auto_tune | original A2C AI | **EnvTune v1.1** |
|---|---|---|---|
| Learning algorithm                  | none (manual UI)        | A2C neural net | SW-UCB1 + Bayesian shrinkage |
| Adaptive to context                 | no                      | yes            | yes (108 contexts)           |
| Risk to WiFi firmware               | none                    | high (removed) | none (no channel-toggling)   |
| CPU footprint                       | tiny                    | high           | tiny (~2-3 % Pi 0 2 W)       |
| Survives reboot                     | manual presets only     | no             | yes (atomic + schema-migrated) |
| Anti-overcapture                    | none                    | partial        | bettercap-skip + per-AP cooldown + tier eviction |
| Fresh-target priority               | none                    | none           | yes (decaying 8-epoch boost) |
| Channel-efficiency feedback         | none                    | none           | yes (multiplier on lifetime score) |
| GPS-zone learning                   | none                    | none           | yes (HS-keyed channel histogram) |
| Web UI                              | parameter editor        | none           | dashboard + 5 CSRF-protected actions |
| Optimises specifically for unique HS| no                      | partly         | yes (60 % weight on lifetime-new/min) |

---

## Troubleshooting

**"Parameters never change"**
First 5 epochs are warmup (observation). After that, UCB only changes a parameter when its current arm has been outperformed in the current context. With sparse data this can take dozens of epochs — that's by design (avoid thrashing). Shrinkage helps cold contexts converge faster but doesn't eliminate this entirely.

**"Plugin uses too much CPU"**
Drop to `cpu_profile = "light"` or `"minimal"`. The big knob is `ucb_window` × `ap_track_max`.

**"State file looks wrong / I want a fresh start"**
Stop pwnagotchi, remove `/etc/pwnagotchi/envtune_state.json`, restart. The captured-BSSID set is rebuilt from `/root/handshakes/` on next load.

**"Logs full of `nexmon crash suspected`"**
Driver instability. EnvTune backs off automatically; if it persists, lower `extra_channels` and `max_interactions`.

**"Web UI buttons go to /plugins/<wrong-name>"**
Fixed in 1.1.0 — earlier versions used relative form actions which broke when the dashboard was visited without a trailing slash. Upgrade.

**"reward EMA shows nonsense like -8e15"**
Fixed in 1.1.0 — input/output of the EMA are now clamped per-key, and the load path drops corrupt persisted values. The next epoch re-seeds cleanly.

---

## Versions

**v1.2.0** *(current)*
- **Annealed empirical-Bayes shrinkage** — shrinkage strength `k` decays from 5 to 1 as the UCB table accumulates real samples (over the first 500). Cold-start cells still inherit the parent prior; mature cells trust their local data. Concrete effect: an arm with `n=13, mean=0.40` had its effective mean dragged down to 0.372 in v1.1; in v1.2 late-game it sits at 0.393. Genuinely-better-but-undertested arms now converge faster.
- **Tier-based GPS zone eviction** — zones with ≥50 attacks and zero handshakes are evicted before never-touched zones, since the untouched ones may still produce captures later. Telemetry showed 12/17 zones in real use sitting at 0 HS with attack counts up to 60+; those now drop out cleanly.
- Three new tunables: `ucb_shrinkage_k_max`, `ucb_shrinkage_k_min`, `ucb_shrinkage_anneal_samples`. Setting `ucb_shrinkage_k` explicitly still pins shrinkage to a fixed value (legacy v1.1 behaviour).

**v1.1.0**
- Empirical-Bayes shrinkage in UCB pick (n/(n+5) blend with parent-group mean) — drastically faster convergence in the 108-state × 14-arm space
- Hill-saturated reward gradient — UCB can finally distinguish target-hit from cold-start prior
- Per-channel efficiency multiplier (0.5×–1.5×)
- Fresh-session BSSID priority bonus (1.5 → 0 over 8 epochs)
- Two new UCB arms: `bored_num_epochs`, `sad_num_epochs`
- Reward gains explicit blind / sad / bored penalties + activity floor
- 25-counter Prometheus endpoint
- Five POST actions with CSRF
- 5 GHz-aware recon time
- Saturation-aware exploration boost
- Handshake-dir + potfile periodic rescans
- EMA input/output clamps + load-time sanitisation (kills the `reward = -8e15` regression)
- Web UI base-href + absolute action paths (fixes the `/plugins/<wrong-prefix>` button bug)
- Tier-based AP eviction; GPS zone LRU cap (500)
- Adaptive `reward_delay` based on AP density
- Immediate bettercap skip-list push on every handshake
- Reward-component breakdown for debug logs

**v1.0.0**
- Initial release: SW-UCB1, hierarchical priors, GPS zones, thermal safety, web UI, async state save.

---

## Credits

Built on prior art by:

- [@evilsocket](https://github.com/evilsocket) — original pwnagotchi
- [@jayofelony](https://github.com/jayofelony) — noai fork
- [@Sniffleupagus](https://github.com/Sniffleupagus) — `auto_tune` plugin
- [@rai68](https://github.com/rai68) + [@AlienMajik](https://github.com/AlienMajik) — TheyLive GPS plugin
- @hasj — earlier `envtune` iterations

---

## License

MIT
