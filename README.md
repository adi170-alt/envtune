<p align="center">
  <img src="https://img.shields.io/badge/Pwnagotchi-Plugin-ff69b4?style=for-the-badge&logo=raspberry-pi&logoColor=white" alt="Pwnagotchi Plugin" />
  <img src="https://img.shields.io/badge/Version-3.0.0-blue?style=for-the-badge" alt="Version 3.0.0" />
  <img src="https://img.shields.io/badge/Author-adi1708-orange?style=for-the-badge" alt="Author adi1708" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License" />
</p>

<h1 align="center">🎛️ EnvTune</h1>

> **An advanced, environment-aware personality tuner for Pwnagotchi.**
> 
> EnvTune actively monitors your surroundings and dynamically adjusts your Pwnagotchi's core parameters (`min_rssi`, `hop_recon_time`, `recon_time`, and `max_interactions`) to optimize handshake capture rates and maximize rewards.

---

## ✨ Key Features

* 📈 **EMA Smoothing:** Uses Exponential Moving Average to smooth out spikes in access point (AP) density, handshake rates, and rewards. This prevents your Pwnagotchi from overreacting to brief changes in the environment.
* 💾 **State Persistence:** Saves lifetime handshakes, top channels, and best-performing settings to disk. Your Pwnagotchi learns and remembers its optimal configurations across reboots!
* 🧠 **Best-Settings Memory:** Once EnvTune finds a configuration that yields high rewards, it actively biases future adjustments toward those proven "best" settings.
* ⏪ **Reward-Revert:** If a recent parameter adjustment causes the reward score to drop significantly, EnvTune automatically detects this and rolls back to the previous stable settings.
* 🙈 **Blind-Panic Mode:** If your Pwnagotchi sees nothing for a specified number of epochs, EnvTune safely resets parameters to highly permissive defaults to help it regain its bearings.
* 🏙️ **Density Adaptation:** Automatically tightens parameters in crowded areas to focus on high-quality targets, while relaxing them in sparse environments to cast a wider net.

---

## 🛠️ Installation

**1. Connect to your Pwnagotchi via SSH:**
```bash
ssh pi@(name of your pwnaogtchi).local
```

**2. Download the plugin:**
Download `envtune.py` and place it in your custom plugins directory (default is usually `/usr/local/share/pwnagotchi/custom-plugins/`).
```bash
wget https://raw.githubusercontent.com/adi170-alt/envtune/main/envtune.py -O /usr/local/share/pwnagotchi/custom-plugins/envtune.py
```

**3. Enable the plugin in your config:**
Open your configuration file:
```bash
sudo nano /etc/pwnagotchi/config.toml
```
Add the following line to enable it:
```toml
main.plugins.envtune.enabled = true
```

**4. Restart the service:**
```bash
sudo systemctl restart pwnagotchi
```

---

## ⚙️ Configuration (Optional)

EnvTune works exceptionally well out of the box using its built-in defaults. However, power users can fine-tune its behavior by adding the following parameters to `config.toml`. 

Here are the default values and what they do:

```toml
# Smoothing factor for EMA (lower = slower reaction to changes)
main.plugins.envtune.ema_alpha = 0.35

# Epochs to wait before EnvTune starts adjusting parameters
main.plugins.envtune.warmup_epochs = 3

# Handshake rate thresholds for tuning hop speeds
main.plugins.envtune.hs_rate_low = 0.15
main.plugins.envtune.hs_rate_high = 0.40

# Access point thresholds for environment density 
main.plugins.envtune.dense_aps = 25
main.plugins.envtune.sparse_aps = 8

# Epochs of zero visibility before triggering a permissive reset
main.plugins.envtune.blind_panic_epochs = 3

# How much the reward must drop to trigger a settings rollback
main.plugins.envtune.reward_drop_threshold = 0.25

# How frequently (in epochs) to save the state to disk
main.plugins.envtune.save_every_n_epochs = 5

# How heavily to bias new adjustments toward historically best settings (0.0 to 1.0)
main.plugins.envtune.best_bias_weight = 0.15
```

---

## 📂 Under the Hood: The State File

EnvTune automatically generates a state file located at:
`/etc/pwnagotchi/envtune_state.json`

**You do not need to manually edit this file.** It is securely used by the plugin to log your Pwnagotchi's lifetime handshakes, channel statistics, and the best configuration parameters it has discovered over time.
