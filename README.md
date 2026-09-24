# Rimth

> Turn a locked-down XiaoAi speaker into a AI voice assistant powered by Groq.

**Rimth** is an open-source voice assistant written in Rust, designed for the **XiaoAi Speaker LX06** (and similar models). By replacing the stock firmware, it connects the speaker to Groq's Whisper STT, GPT-OSS LLM, and Orpheus TTS for low-latency, customizable, privacy-friendly voice interaction.

## Features

- Voice-activated recording using RMS-based VAD, with auto-stop on silence
- Multi-model chat support: `gpt-oss-20b`, `gpt-oss-120b`, `qwen3.8-27b`, `compound`, `compound-mini`
- Unlimited-length TTS via sentence chunking, bypassing the 200-character per-request limit
- Automatic language detection to pick the right TTS model and voice
- AI tool calling: volume control, terminal commands, URL fetching, alarms, LED, sessions, mic gain
- RGB LED state indication: idle, listening, speaking, muted
- Alarm management via voice
- Hardware key handling: mute, volume up/down, play/pause (interrupts playback)
- Conversation mode for wake-word-free continuous interaction
- Startup diagnostics for config, audio, network, and API
- Built-in hardware tests: `./Rimth test mic|speaker|volume|keys|network|groq`

## Architecture

```
Microphone -> cpal -> VAD -> WAV -> Groq Whisper STT
                                        |
                                Groq LLM + tool calling
                                        |
                        Text cleaning / chunking / language detection
                                        |
                              Groq Orpheus TTS
                                        |
                        aplay pipe playback (no disk writes)
                                        |
                                     Speaker
```
## Quick Start
First, you need to flash a patched/compiled firmware that can use SSH, which is the foundation for the following steps, such as duhow/xiaoai-patch. But if you're a beginner, check out the installation method below:
Not recommended to use this project because it's archived and no longer maintained：idootop/open-xiaoai.
It's recommended to turn on Groq's ZDR to ensure privacy.
中国用户请看这里：
【Groq对于国内不提供服务，你需要在Rimth.toml内设置代理，例如接入Shadowsocks(R),但首先你需要在工具内启用允许来自LAN的连接，且创建防火墙规则（取决于电脑决定），配置如10.0.01:1080】
# Firmware Flashing Guide

This guide covers flashing custom firmware to the XiaoAi Speaker LX06 (and L06A) to enable SSH access, which is required before deploying Rimth.

> **Warning:** Flashing custom firmware carries risk of bricking the device and voids the warranty. Proceed at your own risk. Always back up the original firmware first.

---

## Prerequisites

- XiaoAi Speaker LX06 or L06A
- Windows 10/11 PC (or Linux with the alternative tool)
- Micro-USB data cable (must support data transfer, not charge-only)
- Amlogic Flash Tool v6.0.0

For the LX06 (old model), the Micro-USB debug port is **inside the speaker**. You must disassemble the case to access it. The port is located on the mainboard, typically in the upper-left area. For the newer OH2P model, the Type-C port on the bottom is directly accessible without disassembly.

---

## Step 1: Install the Flash Tool

Download Amlogic Flash Tool v6.0.0 from:
```
https://androidmtk.com/download-amlogic-flash-tool
```

Extract the ZIP to a folder, for example `Amlogic_Flash_Tool_v6.0.0`.

Install the USB driver by running `AMLLogic driver installer.exe` from the `drivers` folder. Select both **USB Driver** and **Serial driver**. You may need to allow unsigned driver installation on Windows.

> **Linux alternative:** See the `duhow/xiaoai-patch` Linux guide for using `aml-flash-tool` with `libusb`.

---

## Step 2: Connect and Enter Flash Mode

1. Connect the speaker to your PC using the Micro-USB cable.
2. Navigate to the `bin` folder of the flash tool in a terminal or Git Bash.
3. Prepare the command `./update.exe identify` but **do not press Enter yet**.
4. Power on the speaker. Within approximately 2 seconds, press Enter to execute the command.
5. If successful, you will see the firmware version:
   ```
   This firmware version is 0-7-0-16-0-0-0-0
   ```

If the command fails, power off the speaker, wait a few seconds, and repeat. It may take several attempts.

---

## Step 3: Set Boot Delay and Backup

Set a boot delay so you can interrupt U-Boot for recovery if needed:
(This will set the boot to boot1 to achieve it)

```bash
./update.exe bulkcmd "setenv bootdelay 15"
./update.exe bulkcmd "setenv boot_part boot1"
./update.exe bulkcmd "saveenv"
```

Back up the original partitions to your PC:

```bash
./update.exe mread store bootloader normal 0x200000 mtd0.img
./update.exe mread store tpl normal 0x800000 mtd1.img
./update.exe mread store boot0 normal 0x600000 mtd2.img
./update.exe mread store boot1 normal 0x600000 mtd3.img
./update.exe mread store system0 normal 0x2800000 mtd4.img
./update.exe mread store system1 normal 0x2800000 mtd5.img
./update.exe mread store data normal 0x13e0000 mtd6.img
```

> **Important:** Store these backup images in a safe place. They are your only way to restore the original firmware.

---

## Step 4: Erase the Data Partition

The `/data` partition may contain leftover configuration from the stock firmware. Erase it to ensure a clean start:

```bash
./update.exe bulkcmd "nand erase 0x000006c20000 0x013E0000"
```

This erases only the `data` partition (starting at `0x6c20000`, size `0x13E0000`). Other partitions are not affected.

> **Partition layout for reference:**

| Partition | Start Address | Size |
|:---|:---|:---|
| bootloader | 0x000000000000 | 0x200000 |
| tpl | 0x0008000000 | 0x800000 |
| boot0 | 0x0010000000 | 0x600000 |
| boot1 | 0x0016000000 | 0x600000 |
| system0 | 0x001c000000 | 0x2620000 |
| system1 | 0x0044200000 | 0x2800000 |
| data | 0x006c20000 | 0x13E0000 |

---

## Step 5: Flash the Patched Firmware

Download the patched firmware from the release page. For LX06/L06A, download the file ending with `lx06.tar` and extract it. You will get two files:

- `boot.img` — kernel
- `root.squashfs` — root filesystem
  
For completeness, you can flash the boot partition.
```bash
./update.exe partition boot1 boot.img
```

Flash the root filesystem to the **inactive** `system1` partition:

```bash
./update.exe partition system1 root.squashfs
```

> **Note:** Do **not** flash `boot.img` unless you specifically need to replace the kernel. The stock kernel is sufficient for most use cases.

---

## Step 6: Switch Boot Partition and Reboot

Switch the boot partition to `boot1` (the partition you just flashed):

```bash
./update.exe bulkcmd "setenv boot_part boot1"
./update.exe bulkcmd "saveenv"
```

Then reboot the speaker:

```bash
./update.exe bulkcmd "reset"
```
At this point, the startup will launch boot1 and system1.
---

## Step 7: Verify and Connect via SSH

After the speaker reboots (approximately 30 seconds), find its IP address on your router. Then attempt to connect:

```bash
ssh root@<speaker-ip>
```

For `duhow/xiaoai-patch`, the default password may be `root`.

If SSH connects successfully, you are ready to deploy Rimth.

---

## Recovery: If the Speaker Does Not Boot

If the speaker fails to boot after flashing, you can revert to the original partition:

1. Connect via USB again and enter flash mode.
2. Switch back to the original boot partition:
   ```bash
   ./update.exe bulkcmd "setenv boot_part boot0"
   ./update.exe bulkcmd "saveenv"
   ./update.exe bulkcmd "reset"
   ```

If USB access is not possible, you will need to use the TTL serial console (see below).

---

## TTL Serial Recovery (Alternative Method)

If the speaker is completely unresponsive to USB, connect via TTL serial:

1. Disassemble the speaker and locate the TTL pads on the mainboard (GND, TX, RX).
2. Connect a USB-to-TTL adapter (3.3V) to these pads.
3. Open a serial terminal (e.g., PuTTY) at **115200** baud, 8N1.
4. Power on the speaker and press the spacebar repeatedly to interrupt U-Boot.
5. At the `lx06#` prompt, run:
   ```
   nand erase 0x000006c20000 0x013E0000
   ```
6. Reset the board with `reset`.

---

## Notes on Partition Safety

- Only flash to the **inactive** partition (`system1` when booted from `boot0`).
- Never flash to the `bootloader` or `tpl` partitions unless you have a full recovery image.
- The `data` partition is safe to erase; it will be reinitialized on boot.

---

## Next Steps

Once SSH is confirmed working, proceed to deploy Rimth:

```bash
scp -O target/armv7-unknown-linux-gnueabihf/release/Rimth root@<speaker-ip>:/data/rimth/
scp -O Rimth.toml root@<speaker-ip>:/data/rimth/
ssh root@<speaker-ip> "cd /data/rimth && chmod +x Rimth && ./Rimth"
```

## Requirements

- XiaoAi Speaker LX06 (black, with infrared) or a compatible model
- Disassembled unit with Micro-USB or TTL serial access
- Third-party firmware with SSH enabled (see `duhow/xiaoai-patch`)
- A Linux host for cross-compilation (Fedora recommended)

## Build

```bash
rustup target add armv7-unknown-linux-gnueabihf

# Using cross (recommended, handles ALSA dependencies automatically)
cargo install cross
cross build --release --target armv7-unknown-linux-gnueabihf

# Or using cargo-zigbuild
cargo install cargo-zigbuild
cargo zigbuild --release --target armv7-unknown-linux-gnueabihf
```

The binary is produced at `target/armv7-unknown-linux-gnueabihf/release/Rimth`.

## Deploy

```bash
scp -O target/armv7-unknown-linux-gnueabihf/release/Rimth root@192.168.1.7:/data/rimth/
scp -O Rimth.toml root@192.168.1.7:/data/rimth/

ssh root@192.168.1.7 "cd /data/rimth && chmod +x Rimth && ./Rimth"
```

On first run, `Rimth.toml` is generated automatically. Edit it to set `groq.api_key` and `proxy.url`, then run again.

## Configuration

See [Rimth.toml](Rimth.toml) for the full configuration reference.

## Testing

```bash
./Rimth test mic        # 3-second recording with live level meter
./Rimth test speaker    # 440Hz test tone
./Rimth test volume     # Volume sweep test
./Rimth test keys       # Key code detection
./Rimth test network    # Proxy exit IP and region
./Rimth test groq       # API key, models, TTS
./Rimth test            # Run all tests
```

# Rimth.toml Configuration Notes

## General Rules

- **File location**: Must be placed next to the `Rimth` executable (`/data/rimth/Rimth.toml`). The binary looks for it in its own directory, not the current working directory.
- **Format**: Standard TOML. Strings use double quotes, booleans use `true`/`false`, integers are plain, floats need a decimal point.
- **Comments**: Use `#`. Inline comments after a value are allowed.
- **Restart required**: Configuration is loaded only at startup. Any change requires restarting the process.
- **No BOM, no CRLF issues**: Save as UTF-8 without BOM. Windows editors may add CRLF line endings; prefer LF for the target device.

## `[groq]`

### `api_key`
- **Must be set** for any AI functionality. Empty string disables all STT/LLM/TTS.
- Format: `gsk_` prefix followed by a long string.
- **Never commit this to a public repository.** Add `Rimth.toml` to `.gitignore` and use `Rimth.toml.example` for templates.
- If leaked, revoke it immediately at https://console.groq.com/keys.

### `stt_model`, `llm_model`, `tts_model`
- Must be **exact model IDs** as returned by `https://api.groq.com/openai/v1/models`.
- Case-sensitive.
- The diagnostics section verifies each model against your account's available list.

### `tts_model` special case
- `canopylabs/orpheus-v1-english` and `canopylabs/orpheus-arabic-saudi` require **one-time terms acceptance** at:
  https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english
- If not accepted, TTS returns HTTP 400 `model_terms_required`.

### `tts_voice`
- Only meaningful for Orpheus models.
- Valid voices for `orpheus-v1-english`: `troy`, `hannah`, `autumn`, `diana`, `austin`, `daniel`.
- Invalid voice names produce a TTS API error.

### `language`
- ISO 639-1 code used as a **hint to Whisper STT**, not as a hard constraint.
- The AI is instructed to reply in the user's actual language, not this value.
- Use `en` for English, `zh` for Chinese, etc. Wrong hints slightly reduce STT accuracy but do not break anything.

### `system_prompt`
- Passed to the LLM along with language and model lists. Keep it under ~1500 characters.
- **Multi-line strings**: Use triple quotes if you want line breaks:
  ```toml
  system_prompt = """
  You are Rimth.
  Keep answers short.
  """
  ```
  Single-line strings with `\n` are also fine.
- Avoid quotes inside the string unless escaped (`\"`) or use single quotes (`'...'`).

### `*_models_available` lists
- These are informational for the AI, not enforced by the code.
- The AI can call `switch_model` with any name from these lists.
- If you list an invalid model, the next chat request will fail with `model_not_found`.
- Keep the lists consistent with what your Groq account actually has access to.

## `[audio]`

### `trigger_threshold`
- RMS threshold to start recording. Range `0.0` to `1.0`.
- **Too low** → constant false triggers from ambient noise.
- **Too high** → never triggers.
- Typical: `0.03` (quiet room) to `0.10` (noisy environment).
- The RMS is computed from normalized `f32` samples in `[-1.0, 1.0]`.

### `silence_threshold`
- Must be **strictly less than** `trigger_threshold`. If equal or greater, recordings will never end automatically.
- Typical: `0.005`.

### `silence_duration_ms`
- Milliseconds of continuous silence before recording stops.
- Too short → cuts off mid-sentence.
- Too long → feels unresponsive.
- Recommended: `1500` to `2500`.

### `min_recording_ms`
- Minimum recording length. Anything shorter is discarded before STT.
- Set to at least `800` to filter out button-press clicks.

### `sample_rate`
- Must match what Whisper expects. **16000 Hz is strongly recommended.**
- Other values (22050, 44100) cause Whisper to resample internally, but 16k avoids the overhead.
- Do **not** change to 48000 — it doubles audio size for no benefit.

### `mixer_control` and `mic_mixer_control`
- ALSA mixer control names, verified with:
  ```bash
  amixer -c 0 scontrols
  ```
- On LX06, the correct name is **`mysoftvol`** (not `Master` or `PCM`).
- If wrong, volume changes silently fail and only print a warning.

### `max_volume`
- **Important hardware protection.** Values above 80 risk damaging the small speaker.
- Range `0-100`. Applied as an upper clamp on every `set_volume` call.

### `card_index`
- ALSA card number. Almost always `0` on LX06.
- Verify with `cat /proc/asound/cards`.

### `mic_gain`
- Software multiplier applied to mic samples in the cpal callback.
- Range: `0.1` to `10.0`. Default `1.0` (no amplification).
- Values above `3.0` may clip loud speech; values below `0.5` make speech inaudible.
- AI can override this at runtime via `set_mic_gain`.

## `[gpio]`

- **Legacy section.** LX06 does not use GPIO for the mute button.
- Always keep `enabled = false` on LX06.
- Only enable this on older models that expose a sysfs GPIO for mute.

## `[keys]`

### `enabled`
- Set to `false` to disable hardware key handling entirely.

### `device`
- The input device node. For LX06, `/dev/input/event0` is correct.
- Verify with `cat /proc/bus/input/devices`.
- If the file doesn't exist, the key listener silently exits.

### `mute`, `volume_up`, `volume_down`, `play_pause`
- Linux key codes (not ASCII). **Must match the actual hardware**.
- Wrong codes cause the wrong action to trigger (e.g. pressing vol+ mutes the mic).
- Verify with `./Rimth test keys` and press each button.

### `volume_step`
- Percentage change per vol+ / vol- press. Range `1-20`.
- Higher values feel jumpy; lower values require many presses.

## `[proxy]`

### `url`
- Empty string = direct connection.
- **Required for users in mainland China and Hong Kong.** Groq blocks those regions.
- Supported schemes:
  - `http://host:port` — HTTP CONNECT proxy
  - `socks5://host:port` — SOCKS5 with local DNS
  - `socks5h://host:port` — SOCKS5 with remote DNS (more reliable when local DNS is poisoned)
- **Do not include a trailing slash.**
- Example: `url = "http://192.168.1.133:1080"`

### Common pitfalls
- If the proxy is on another machine, that machine must allow LAN connections (bind to `0.0.0.0`, not `127.0.0.1`) and its firewall must permit the port.
- A running `kaspersky` or similar security suite can silently block LAN traffic to the proxy port.

## `[session]`

### `persist_dir`
- Empty string → defaults to `<exe_dir>/sessions/`.
- Set to an absolute path to store sessions elsewhere.

### `max_history`
- Maximum number of messages kept per session (system prompt excluded).
- **Lower values reduce token usage and avoid TPM rate limits.**
- Groq free tier `openai/gpt-oss-20b` has **8000 TPM limit**. With history plus tools plus system prompt, each request consumes 3000-4500 tokens.
- Recommended: **`20` to `30`**. Values above `50` risk frequent 429 errors.
- Old messages are dropped from the front when the limit is exceeded.

## `[diagnostics]`

### `run_on_startup`
- When `true`, runs a full check on every launch. Adds ~5-10 seconds to startup.
- Set `false` for faster boot once you've verified everything works.

### `test_tts_on_startup`
- When `true`, also performs a TTS synthesis round-trip during diagnostics. **Consumes API credits.**
- Leave `false` unless debugging TTS specifically.

## `[led]`

### `enabled`
- Set `false` to disable LED control entirely (LED worker won't start).

### `device`
- **Must be** `/sys/devices/i2c-0/0-003a/led_rgb` on LX06.
- Path `/sys/class/leds/xiaomi:rgb:status` does not exist on LX06 and will cause the LED worker to disable itself.
- Verify with:
  ```bash
  ls /sys/devices/i2c-0/0-003a/
  ```

### `brightness`
- Range `0-100`. Applied as a software multiplier on top of every frame.
- **Does not touch the hardware `led_imax` register.**
- Values above `70` may be uncomfortably bright at night.

### Color fields (`idle_color`, `listening_color`, `speaking_color`, `muted_color`, `thinking_color`)
- Format: `RRGGBB` hex, **no `#` prefix, no `0x`**.
- Case-insensitive (`FF00AA` and `ff00aa` are the same).
- The code converts to BGR internally for the AW20054 driver.
- Colors that are too saturated will look washed out due to the brightness multiplier.

### `thinking_color`
- Only used during the rotating animation while the LLM is generating a reply.
- A bright purple or warm orange works well visually.

## `[alarm]`

### `store_path`
- Path to the JSON file where alarms are persisted.
- The parent directory is created automatically if missing.
- Format: a JSON array of objects with `id`, `time`, `label`, `repeat`, `enabled`.
- Safe to edit manually while the program is not running.
- Falsy IDs (`alarm_1`, `alarm_2`, ...) are assigned automatically.

## Common Failures Checklist

| Symptom | Likely cause |
|:---|:---|
| `Permission denied` on config | Wrong owner or mode; chmod to 644 |
| `Parse error` | Missing quote, stray comma, CRLF line endings |
| LED writes fail silently | `led.device` path wrong or file is read-only |
| Volume changes do nothing | `mixer_control` name wrong |
| STT always returns empty | `trigger_threshold` too high, or mic gain 0 |
| `429 rate_limit_exceeded` | `max_history` too high, or `max_tokens` too large |
| `400 Tools should have a name` | Corrupted session history — delete `sessions/*.json` |
| `model_terms_required` | TTS terms not accepted at Groq console |
| Proxy connects but API fails | Wrong scheme (`http` vs `socks5`), or DNS resolution issue |
| Alarm doesn't fire | Device timezone wrong — run `date` and fix `/etc/timezone` |

## Security Reminders

- **Never commit `Rimth.toml`** with a real `api_key`. Add it to `.gitignore`.
- The `run_terminal` tool allows arbitrary shell execution. Consider restricting it in production.
- If using a proxy, ensure it's on a trusted network segment.
- Back up `sessions/` and `alarms.json` before major upgrades.

## Usage Examples

- "Set volume to 30"
- "Switch to a smarter model"
- "Set an alarm for 7 AM"
- "Turn the LED purple"
- "Enter conversation mode"
- "Start a new session"
- "What's the weather in Beijing today"

## Disclaimer

- Flashing custom firmware carries risk of bricking the device. Back up the original firmware first.
- Groq's Terms of Service prohibit bypassing geographic restrictions via proxy or VPN. Users in mainland China and Hong Kong assume account risk when using a proxy.
- Canopy Labs' TTS Terms of Service prohibit reverse engineering, weight extraction, or commercial use.
- The `run_terminal` tool allows the AI to execute arbitrary shell commands. Use with caution.

## License

AGPL

## Acknowledgements

- [duhow/xiaoai-patch](https://github.com/duhow/xiaoai-patch) - XiaoAi speaker patch firmware
- [Groq](https://groq.com/) - Ultra-low-latency inference platform
- [cpal](https://github.com/RustAudio/cpal), [hound](https://github.com/ruuda/hound), [reqwest](https://github.com/seanmonstar/reqwest)

Awa - zhdljc|Tarn - \
