// Rimth - Voice Assistant for XiaoAi Speaker (LX06)
// Copyright (C) 2026 zhdljc
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use hound::{SampleFormat as HoundSampleFormat, WavReader, WavSpec, WavWriter};
use reqwest::multipart;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex as StdMutex};
use std::time::Duration;
use tokio::sync::mpsc::{channel, unbounded_channel, Receiver, Sender, UnboundedReceiver, UnboundedSender};
use tokio::sync::RwLock;
use tracing::{error, info, warn};

const STT_URL: &str = "https://api.groq.com/openai/v1/audio/transcriptions";
const TTS_URL: &str = "https://api.groq.com/openai/v1/audio/speech";
const CHAT_URL: &str = "https://api.groq.com/openai/v1/chat/completions";
const MODELS_URL: &str = "https://api.groq.com/openai/v1/models";
const IP_API_URL: &str = "http://ip-api.com/json/?fields=status,country,countryCode,regionName,city,query,timezone";
const CONFIG_FILE: &str = "Rimth.toml";

const TTS_CHUNK_CHARS: usize = 180;
const TTS_PREFETCH: usize = 2;
const MIN_AUDIO_BYTES: usize = 44 + 16000 * 2 * 1;
const LED_COUNT: usize = 18;
const LED_FPS_MS: u64 = 33;
const LED_MAX_CONSECUTIVE_FAILURES: u32 = 10;
const LED_RECOVERY_MS: u64 = 5000;
const CAPTURE_IDLE_TIMEOUT_SECS: u64 = 5;

type SharedGroq = Arc<RwLock<GroqConfig>>;

static OUTPUT_LOCK: LazyLock<StdMutex<()>> = LazyLock::new(|| StdMutex::new(()));

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    groq: GroqConfig,
    audio: AudioConfig,
    gpio: GpioConfig,
    keys: KeysConfig,
    proxy: ProxyConfig,
    session: SessionConfig,
    diagnostics: DiagnosticsConfig,
    led: LedConfig,
    alarm: AlarmConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GroqConfig {
    api_key: String,
    stt_model: String,
    llm_model: String,
    tts_model: String,
    tts_voice: String,
    language: String,
    system_prompt: String,
    #[serde(default)]
    stt_models_available: Vec<String>,
    #[serde(default)]
    llm_models_available: Vec<String>,
    #[serde(default)]
    tts_models_available: Vec<String>,
    #[serde(default)]
    tts_voices_available: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AudioConfig {
    trigger_threshold: f32,
    silence_threshold: f32,
    silence_duration_ms: u32,
    min_recording_ms: u32,
    sample_rate: u32,
    mixer_control: String,
    mic_mixer_control: String,
    max_volume: u32,
    card_index: u32,
    mic_gain: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GpioConfig { mute_pin: u64, enabled: bool }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KeysConfig {
    enabled: bool,
    device: String,
    mute: u16,
    volume_up: u16,
    volume_down: u16,
    play_pause: u16,
    volume_step: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProxyConfig { url: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionConfig { persist_dir: String, max_history: usize }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiagnosticsConfig { run_on_startup: bool, test_tts_on_startup: bool }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LedConfig {
    enabled: bool,
    #[serde(default = "default_led_device")]
    device: String,
    #[serde(default = "default_led_brightness")]
    brightness: u8,
    #[serde(default = "default_idle_color")]
    idle_color: String,
    #[serde(default = "default_listening_color")]
    listening_color: String,
    #[serde(default = "default_speaking_color")]
    speaking_color: String,
    #[serde(default = "default_muted_color")]
    muted_color: String,
    #[serde(default = "default_thinking_color")]
    thinking_color: String,
}

fn default_led_device() -> String { "/sys/devices/i2c-0/0-003a/led_rgb".into() }
fn default_led_brightness() -> u8 { 60 }
fn default_idle_color() -> String { "0000C8".into() }
fn default_listening_color() -> String { "00C800".into() }
fn default_speaking_color() -> String { "C80000".into() }
fn default_muted_color() -> String { "C800C8".into() }
fn default_thinking_color() -> String { "8000FF".into() }

impl Default for LedConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device: default_led_device(),
            brightness: default_led_brightness(),
            idle_color: default_idle_color(),
            listening_color: default_listening_color(),
            speaking_color: default_speaking_color(),
            muted_color: default_muted_color(),
            thinking_color: default_thinking_color(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AlarmConfig { store_path: String }

impl Default for Config {
    fn default() -> Self {
        Self {
            groq: GroqConfig {
                api_key: String::new(),
                stt_model: "whisper-large-v3-turbo".into(),
                llm_model: "openai/gpt-oss-20b".into(),
                tts_model: "canopylabs/orpheus-v1-english".into(),
                tts_voice: "autumn".into(),
                language: "en".into(),
                system_prompt: "You are Rimth, a witty voice assistant on a speaker. Keep answers concise and conversational, with a light touch of humor. You can control volume, run terminal commands, fetch URLs, control the LED, set alarms, switch models, read your own conversation history, tell the time, report system info, stop your own speech, change the response language, and clear the session. Always reply in the user's language. Output plain prose only, no markdown, no tables, no emojis, because the reply will be spoken via TTS. If the user wants multiple alarms at different times, call set_alarm once per time.".into(),
                stt_models_available: vec!["whisper-large-v3-turbo".into(), "whisper-large-v3".into()],
                llm_models_available: vec![
                    "openai/gpt-oss-20b".into(), "openai/gpt-oss-120b".into(),
                    "qwen/qwen3.8-27b".into(), "groq/compound".into(), "groq/compound-mini".into(),
                ],
                tts_models_available: vec![
                    "canopylabs/orpheus-v1-english".into(), "canopylabs/orpheus-arabic-saudi".into(),
                ],
                tts_voices_available: vec![
                    "troy".into(), "hannah".into(), "autumn".into(),
                    "diana".into(), "austin".into(), "daniel".into(),
                ],
            },
            audio: AudioConfig {
                trigger_threshold: 0.05, silence_threshold: 0.005,
                silence_duration_ms: 2000, min_recording_ms: 1000,
                sample_rate: 16000, mixer_control: "mysoftvol".into(),
                mic_mixer_control: "mysoftvol".into(), max_volume: 80,
                card_index: 0, mic_gain: 1.0,
            },
            gpio: GpioConfig { mute_pin: 0, enabled: false },
            keys: KeysConfig {
                enabled: true, device: "/dev/input/event0".into(),
                mute: 102, volume_up: 139, volume_down: 115, play_pause: 114,
                volume_step: 5,
            },
            proxy: ProxyConfig { url: String::new() },
            session: SessionConfig { persist_dir: String::new(), max_history: 30 },
            diagnostics: DiagnosticsConfig { run_on_startup: true, test_tts_on_startup: false },
            led: LedConfig::default(),
            alarm: AlarmConfig { store_path: "/data/rimth/alarms.json".into() },
        }
    }
}

// ---------------------------------------------------------------------------
// Chat data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

impl ChatMessage {
    fn system(content: String) -> Self { Self { role: "system".into(), content: Some(content), tool_calls: None, tool_call_id: None, name: None } }
    fn user(content: String) -> Self { Self { role: "user".into(), content: Some(content), tool_calls: None, tool_call_id: None, name: None } }
    fn assistant(content: Option<String>, tool_calls: Option<Vec<ToolCall>>) -> Self { Self { role: "assistant".into(), content, tool_calls, tool_call_id: None, name: None } }
    fn tool(tool_call_id: String, name: String, content: String) -> Self { Self { role: "tool".into(), content: Some(content), tool_calls: None, tool_call_id: Some(tool_call_id), name: Some(name) } }
}

#[derive(Debug, Deserialize)]
struct SttResponse { text: String }

#[derive(Debug, Deserialize)]
struct ChatResponse { choices: Vec<ChatChoice> }

#[derive(Debug, Deserialize)]
struct ChatChoice { message: ChatMessageResponse }

#[derive(Debug, Deserialize)]
struct ChatMessageResponse { content: Option<String>, tool_calls: Option<Vec<ToolCall>> }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolCall { id: String, #[serde(rename = "type")] call_type: String, function: FunctionCall }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FunctionCall { name: String, arguments: String }

#[derive(Debug, Deserialize)]
struct ModelsResponse { data: Vec<ModelInfo> }

#[derive(Debug, Deserialize)]
struct ModelInfo { id: String }

#[derive(Debug, Deserialize)]
struct IpApiResponse {
    #[allow(dead_code)] status: String,
    country: Option<String>,
    #[serde(rename = "countryCode")] country_code: Option<String>,
    #[serde(rename = "regionName")] region_name: Option<String>,
    city: Option<String>, query: Option<String>, timezone: Option<String>,
}

struct SttError { status: u16, message: String }

enum Prompt { Auth, RateLimit, BadRequest, Server, Network }

enum DiagnosticResult { Ok(String), Warn(String), Fail(String) }

impl DiagnosticResult {
    fn print(&self, label: &str) {
        let tag = match self { DiagnosticResult::Ok(_) => "OK", DiagnosticResult::Warn(_) => "WARN", DiagnosticResult::Fail(_) => "FAIL" };
        let msg = match self { DiagnosticResult::Ok(m) | DiagnosticResult::Warn(m) | DiagnosticResult::Fail(m) => m };
        println!("  [{:4}] {:<28} {}", tag, label, msg);
    }
}

// ---------------------------------------------------------------------------
// LED
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LedState { Idle, Listening, Speaking, Muted, Alarm }

#[derive(Debug)]
enum LedCommand {
    SetState(LedState),
    StartThinking,
    StopThinking,
    SetBrightness(u8),
    SetColorHex(String),
    SetOneHex(usize, String),
    Off,
}

#[derive(Clone)]
struct LedHandle { tx: Option<UnboundedSender<LedCommand>> }

impl LedHandle {
    fn send(&self, cmd: LedCommand) { if let Some(tx) = &self.tx { let _ = tx.send(cmd); } }
}

fn parse_rgb_hex_to_bgr(s: &str) -> Option<u32> {
    let s = s.trim().trim_start_matches('#');
    let v = u32::from_str_radix(s, 16).ok()?;
    let r = (v >> 16) & 0xFF;
    let g = (v >> 8) & 0xFF;
    let b = v & 0xFF;
    Some((b << 16) | (g << 8) | r)
}

/// The AW20054 sysfs driver processes exactly one LED per write syscall.
fn write_leds(path: &str, colors: &[u32; LED_COUNT]) -> Result<()> {
    for (i, c) in colors.iter().enumerate() {
        let payload = format!("{} 0x{:06X}\n", i, c);
        std::fs::write(path, payload).with_context(|| format!("LED write failed at index {}", i))?;
    }
    Ok(())
}

fn step_toward(cur: u32, tgt: u32, step: u8) -> u32 {
    let s = step as i32;
    let cb = ((cur >> 16) & 0xFF) as i32;
    let cg = ((cur >> 8) & 0xFF) as i32;
    let cr = (cur & 0xFF) as i32;
    let tb = ((tgt >> 16) & 0xFF) as i32;
    let tg = ((tgt >> 8) & 0xFF) as i32;
    let tr = (tgt & 0xFF) as i32;
    let nb = if (tb - cb).abs() <= s { tb } else { cb + (tb - cb).signum() * s };
    let ng = if (tg - cg).abs() <= s { tg } else { cg + (tg - cg).signum() * s };
    let nr = if (tr - cr).abs() <= s { tr } else { cr + (tr - cr).signum() * s };
    ((nb as u32) << 16) | ((ng as u32) << 8) | (nr as u32)
}

fn state_to_color(state: LedState, cfg: &LedConfig) -> u32 {
    let hex = match state {
        LedState::Idle => &cfg.idle_color,
        LedState::Listening => &cfg.listening_color,
        LedState::Speaking => &cfg.speaking_color,
        LedState::Muted => &cfg.muted_color,
        LedState::Alarm => &cfg.speaking_color,
    };
    parse_rgb_hex_to_bgr(hex).unwrap_or(0)
}

fn apply_brightness(c: u32, brightness: u8) -> u32 {
    let scale = brightness as u32;
    let b = ((c >> 16) & 0xFF) * scale / 100;
    let g = ((c >> 8) & 0xFF) * scale / 100;
    let r = (c & 0xFF) * scale / 100;
    (b << 16) | (g << 8) | r
}

async fn led_worker(mut rx: UnboundedReceiver<LedCommand>, config: LedConfig) {
    if !config.enabled {
        info!("LED worker disabled");
        while rx.recv().await.is_some() {}
        return;
    }
    if !PathBuf::from(&config.device).exists() {
        warn!("LED device {} not found, disabling LED worker", config.device);
        while rx.recv().await.is_some() {}
        return;
    }

    let mut target = [0u32; LED_COUNT];
    let mut current = [0u32; LED_COUNT];
    let mut brightness = config.brightness.min(100);
    let mut current_state = LedState::Idle;
    let mut thinking = false;
    let mut thinking_phase: u32 = 0;
    let mut consecutive_failures: u32 = 0;
    let mut skip_until: Option<std::time::Instant> = None;

    let idle_bgr = state_to_color(LedState::Idle, &config);
    for i in 0..LED_COUNT { target[i] = idle_bgr; }

    let mut interval = tokio::time::interval(Duration::from_millis(LED_FPS_MS));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    info!("LED worker started on {}", config.device);

    loop {
        tokio::select! {
            biased;
            cmd = rx.recv() => {
                match cmd {
                    Some(LedCommand::SetState(s)) => {
                        thinking = false;
                        current_state = s;
                        let c = state_to_color(s, &config);
                        for i in 0..LED_COUNT { target[i] = c; }
                    }
                    Some(LedCommand::StartThinking) => { thinking = true; thinking_phase = 0; }
                    Some(LedCommand::StopThinking) => {
                        thinking = false;
                        let c = state_to_color(current_state, &config);
                        for i in 0..LED_COUNT { target[i] = c; }
                    }
                    Some(LedCommand::SetBrightness(b)) => { brightness = b.min(100); }
                    Some(LedCommand::SetColorHex(hex)) => {
                        if let Some(bgr) = parse_rgb_hex_to_bgr(&hex) {
                            thinking = false;
                            for i in 0..LED_COUNT { target[i] = bgr; }
                        }
                    }
                    Some(LedCommand::SetOneHex(idx, hex)) => {
                        if idx < LED_COUNT {
                            if let Some(bgr) = parse_rgb_hex_to_bgr(&hex) { target[idx] = bgr; }
                        }
                    }
                    Some(LedCommand::Off) => {
                        thinking = false;
                        for i in 0..LED_COUNT { target[i] = 0; }
                    }
                    None => break,
                }
            }
            _ = interval.tick() => {
                if let Some(until) = skip_until {
                    if std::time::Instant::now() < until { continue; }
                    skip_until = None;
                    consecutive_failures = 0;
                }

                if thinking {
                    thinking_phase = thinking_phase.wrapping_add(1);
                    let head = ((thinking_phase / 6) % LED_COUNT as u32) as i32;
                    let think_bgr = parse_rgb_hex_to_bgr(&config.thinking_color).unwrap_or(0x8000FF);
                    for i in 0..LED_COUNT {
                        let dist = (i as i32 - head).rem_euclid(LED_COUNT as i32);
                        let dist = if dist > (LED_COUNT as i32) / 2 { LED_COUNT as i32 - dist } else { dist };
                        let intensity: u32 = match dist { 0 => 255, 1 => 170, 2 => 100, 3 => 55, 4 => 30, _ => 12 };
                        let factor = intensity * 100 / 255;
                        target[i] = apply_brightness(think_bgr, factor.min(100) as u8);
                    }
                }
                let mut changed = false;
                for i in 0..LED_COUNT {
                    if current[i] != target[i] {
                        current[i] = step_toward(current[i], target[i], 18);
                        changed = true;
                    }
                }
                if changed || thinking {
                    let mut out = [0u32; LED_COUNT];
                    for i in 0..LED_COUNT {
                        out[i] = if thinking { current[i] } else { apply_brightness(current[i], brightness) };
                    }
                    match write_leds(&config.device, &out) {
                        Ok(_) => { consecutive_failures = 0; }
                        Err(e) => {
                            consecutive_failures += 1;
                            if consecutive_failures == 1 { warn!("LED write failed: {}", e); }
                            if consecutive_failures >= LED_MAX_CONSECUTIVE_FAILURES {
                                warn!("LED temporarily disabled after {} failures, retrying in {} ms",
                                      consecutive_failures, LED_RECOVERY_MS);
                                skip_until = Some(std::time::Instant::now() + Duration::from_millis(LED_RECOVERY_MS));
                            }
                        }
                    }
                }
            }
        }
    }
    info!("LED worker stopped");
}

fn spawn_led_worker(config: LedConfig) -> LedHandle {
    if !config.enabled { return LedHandle { tx: None }; }
    let (tx, rx) = unbounded_channel();
    tokio::spawn(led_worker(rx, config));
    LedHandle { tx: Some(tx) }
}

// ---------------------------------------------------------------------------
// Capture context
// ---------------------------------------------------------------------------

struct CaptureCtx {
    inner: Arc<StdMutex<Option<Stream>>>,
    tx: Sender<Vec<f32>>,
    sample_rate: u32,
    mic_gain: Arc<AtomicU32>,
}

impl CaptureCtx {
    fn new(tx: Sender<Vec<f32>>, sample_rate: u32, mic_gain: Arc<AtomicU32>) -> Result<Self> {
        let stream = start_capture(tx.clone(), sample_rate, mic_gain.clone())?;
        Ok(Self { inner: Arc::new(StdMutex::new(Some(stream))), tx, sample_rate, mic_gain })
    }

    /// Drop the current input stream and create a fresh one. Used only as a
    /// watchdog fallback: with cpal output the input side is not disturbed by
    /// playback, so this normally never has to run.
    fn restart(&self) -> Result<()> {
        { let mut g = self.inner.lock().unwrap(); *g = None; }
        let mut last_err: Option<anyhow::Error> = None;
        for delay_ms in [80u64, 160, 320, 640] {
            std::thread::sleep(Duration::from_millis(delay_ms));
            match start_capture(self.tx.clone(), self.sample_rate, self.mic_gain.clone()) {
                Ok(s) => {
                    let mut g = self.inner.lock().unwrap();
                    *g = Some(s);
                    info!("Capture stream restarted");
                    return Ok(());
                }
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("Capture restart failed")))
    }
}

// ---------------------------------------------------------------------------
// Paths / utilities
// ---------------------------------------------------------------------------

fn exe_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("Failed to get current executable path")?;
    Ok(exe.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(".")))
}

fn config_path() -> Result<PathBuf> { Ok(exe_dir()?.join(CONFIG_FILE)) }
fn sounds_dir() -> Result<PathBuf> { Ok(exe_dir()?.join("sounds")) }

fn sessions_dir() -> Result<PathBuf> {
    let dir = exe_dir()?.join("sessions");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn build_client(proxy_url: &str) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .connect_timeout(Duration::from_secs(15));
    if !proxy_url.is_empty() {
        let p = reqwest::Proxy::all(proxy_url)?;
        builder = builder.proxy(p);
    }
    Ok(builder.build()?)
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() { return 0.0; }
    let sum: f32 = samples.iter().map(|&s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

fn parse_volume(stdout: &str) -> Option<u32> {
    let start = stdout.find('[')?;
    let end = stdout[start..].find('%')?;
    stdout[start + 1..start + end].parse().ok()
}

fn local_time_hm() -> Option<String> {
    std::process::Command::new("date").arg("+%H:%M").output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok()).map(|s| s.trim().to_string())
}
fn local_date() -> Option<String> {
    std::process::Command::new("date").arg("+%Y-%m-%d").output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok()).map(|s| s.trim().to_string())
}
fn local_weekday() -> u8 {
    std::process::Command::new("date").arg("+%u").output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u8>().ok()).unwrap_or(0)
}
fn local_time_hms() -> String {
    std::process::Command::new("date").arg("+%Y-%m-%d %H:%M:%S %Z").output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok()).map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

// ---------------------------------------------------------------------------
// Alarms
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Alarm { id: String, time: String, label: String, repeat: String, enabled: bool }

fn load_alarms(path: &str) -> Vec<Alarm> {
    if let Ok(content) = std::fs::read_to_string(path) {
        if let Ok(alarms) = serde_json::from_str::<Vec<Alarm>>(&content) { return alarms; }
    }
    Vec::new()
}

fn save_alarms(path: &str, alarms: &[Alarm]) {
    if let Some(parent) = PathBuf::from(path).parent() { let _ = std::fs::create_dir_all(parent); }
    if let Ok(content) = serde_json::to_string_pretty(alarms) { let _ = std::fs::write(path, content); }
}

fn add_alarm(path: &str, time: &str, label: &str, repeat: &str) -> String {
    let mut alarms = load_alarms(path);
    let mut n = 1;
    loop {
        let candidate = format!("alarm_{}", n);
        if !alarms.iter().any(|a| a.id == candidate) {
            alarms.push(Alarm { id: candidate.clone(), time: time.to_string(), label: label.to_string(), repeat: repeat.to_string(), enabled: true });
            save_alarms(path, &alarms);
            return format!("Alarm set for {} ({}, id {})", time, label, candidate);
        }
        n += 1;
    }
}

fn delete_alarm(path: &str, id: &str) -> String {
    let mut alarms = load_alarms(path);
    let before = alarms.len();
    alarms.retain(|a| a.id != id);
    if alarms.len() < before { save_alarms(path, &alarms); format!("Alarm {} deleted", id) }
    else { format!("Alarm {} not found", id) }
}

fn list_alarms_text(path: &str) -> String {
    let alarms = load_alarms(path);
    if alarms.is_empty() { return "No alarms set".into(); }
    alarms.iter().map(|a| format!("{}: {} {} ({})", a.id, a.time, a.label, a.repeat)).collect::<Vec<_>>().join("; ")
}

// ---------------------------------------------------------------------------
// Text processing
// ---------------------------------------------------------------------------

fn strip_markdown(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' | '`' | '~' | '_' => { while let Some(&n) = chars.peek() { if n == c { chars.next(); } else { break; } } }
            '#' => {
                if out.ends_with('\n') || out.is_empty() {
                    while let Some(&n) = chars.peek() { if n == '#' || n == ' ' { chars.next(); } else { break; } }
                }
            }
            '|' => out.push(' '),
            '-' => {
                let mut count = 1;
                while let Some(&'-') = chars.peek() { chars.next(); count += 1; }
                if count >= 3 { out.push(' '); } else { for _ in 0..count { out.push('-'); } }
            }
            '\r' => {}
            _ => out.push(c),
        }
    }
    let mut collapsed = String::with_capacity(out.len());
    let mut last_space = false;
    for c in out.chars() {
        if c == ' ' { if !last_space { collapsed.push(' '); } last_space = true; }
        else { collapsed.push(c); last_space = false; }
    }
    collapsed.trim().to_string()
}

/// Split long text into TTS-safe segments. Uses byte offsets tracked through
/// the growing buffer so multi-byte characters never corrupt the boundary.
fn split_text_for_tts(text: &str, max_chars: usize) -> Vec<String> {
    let mut segments: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_chars = 0usize;
    let mut last_cut_byte: usize = 0;

    for c in text.chars() {
        current.push(c);
        current_chars += 1;
        if matches!(c, '。' | '！' | '？' | '.' | '!' | '?' | ';' | '；' | '\n' | '，' | ',' | ' ') {
            last_cut_byte = current.len();
        }
        if current_chars >= max_chars {
            let head: String;
            let tail: String;
            if last_cut_byte > 0 && last_cut_byte < current.len() {
                head = current[..last_cut_byte].trim().to_string();
                tail = current[last_cut_byte..].trim_start().to_string();
            } else {
                head = current.trim().to_string();
                tail = String::new();
            }
            if !head.is_empty() { segments.push(head); }
            current = tail;
            current_chars = current.chars().count();
            last_cut_byte = 0;
        }
    }
    let tail = current.trim().to_string();
    if !tail.is_empty() { segments.push(tail); }
    if segments.is_empty() && !text.trim().is_empty() {
        segments.push(text.trim().to_string());
    }
    segments
}

fn is_hallucination(text: &str) -> bool {
    let lower = text.to_lowercase();
    let patterns = [
        "thanks for watching", "thank you for watching", "please subscribe", "subscribe to",
        "amara.org", "subtitle", "subtitles", "mbc", "kbs", "jtbc", "yoyo television", "mingshi",
    ];
    if patterns.iter().any(|p| lower.contains(p)) { return true; }
    // CJK hallucination patterns
    let cjk_patterns = [
        "字幕由", "字幕组", "感谢观看", "请订阅", "订阅频道", "感谢大家观看",
        "ご視聴ありがとう", "チャンネル登録", "ご視聴ありがとうございました",
        "시청해주셔서 감사", "구독과 좋아요",
    ];
    cjk_patterns.iter().any(|p| text.contains(p))
}

fn detect_language(text: &str) -> &'static str {
    let mut cjk = 0usize; let mut latin = 0usize; let mut cyrillic = 0usize; let mut arabic = 0usize;
    for c in text.chars() {
        if c >= '\u{4e00}' && c <= '\u{9fff}' { cjk += 1; }
        else if c.is_ascii_alphabetic() { latin += 1; }
        else if c >= '\u{0400}' && c <= '\u{04ff}' { cyrillic += 1; }
        else if c >= '\u{0600}' && c <= '\u{06ff}' { arabic += 1; }
    }
    if cjk > latin && cjk > cyrillic && cjk > arabic { "zh" }
    else if cyrillic > latin && cyrillic > cjk { "ru" }
    else if arabic > latin && arabic > cjk { "ar" }
    else { "en" }
}

fn choose_tts_for_language(lang: &str, config: &GroqConfig) -> (String, String) {
    match lang {
        "ar" => ("canopylabs/orpheus-arabic-saudi".into(), "default".into()),
        _ => {
            let voice = if config.tts_voices_available.is_empty() { "autumn".into() } else { config.tts_voice.clone() };
            ("canopylabs/orpheus-v1-english".into(), voice)
        }
    }
}

fn language_name(code: &str) -> &'static str {
    match code {
        "zh" => "Chinese (Simplified)", "en" => "English", "ja" => "Japanese",
        "ko" => "Korean", "fr" => "French", "de" => "German", "es" => "Spanish",
        "ru" => "Russian", "ar" => "Arabic", "pt" => "Portuguese", "it" => "Italian",
        "nl" => "Dutch", "pl" => "Polish", "tr" => "Turkish", "hi" => "Hindi",
        "th" => "Thai", "vi" => "Vietnamese", "id" => "Indonesian",
        _ => "the user's language",
    }
}

fn build_system_prompt(user_prompt: &str, language: &str, config: &GroqConfig) -> String {
    let lang_name = language_name(language);
    let stt_list = config.stt_models_available.join(", ");
    let llm_list = config.llm_models_available.join(", ");
    let tts_list = config.tts_models_available.join(", ");
    let voice_list = config.tts_voices_available.join(", ");
    format!(
        "{}\n\nAlways reply in {}.\nDetect the language of the user's latest message and respond in that language, even if it differs from the configured language above.\nOutput plain prose only. Do NOT use markdown, tables, bullet points, headings, emojis, or code blocks, because the reply will be spoken through a text-to-speech engine.\nKeep the entire reply under 400 characters when possible.\n\nAvailable models you can switch to using the switch_model tool:\nSTT models: {}\nLLM models: {}\nTTS models: {}\nTTS voices: {}\n\nWhen the user asks to change a model, use the switch_model tool with the appropriate model_type (stt, llm, tts, voice) and model name. Do not just describe the change; actually call the tool.\nWhen the user asks to set an alarm, use the set_alarm tool. If the user wants multiple alarms at different times, call set_alarm once per time.\nWhen the user asks to control the LED, use the led_control tool.\nWhen the user wants to start a conversation mode (no wake word needed), use the conversation_mode tool with enabled=true. To exit, use enabled=false.\nWhen the user asks for a new session, use the session_control tool with action=new.\nWhen the user asks to mute or unmute the microphone, use the set_mute tool.\nWhen the user asks about what was said earlier, use the query_history tool.\nWhen the user asks the time or date, use the get_time tool.\nWhen the user asks for system status, use the get_system_info tool.\nWhen the user asks you to stop talking, use the stop_speaking tool.\nWhen the user asks you to respond in another language, use the set_language tool.\nWhen the user asks you to forget the conversation, use the clear_history tool.",
        user_prompt, lang_name, stt_list, llm_list, tts_list, voice_list
    )
}

// ---------------------------------------------------------------------------
// Key listener
// ---------------------------------------------------------------------------

fn adjust_volume_relative(config: &AudioConfig, delta: i32) {
    let output = std::process::Command::new("amixer")
        .args(["-c", &config.card_index.to_string(), "get", &config.mixer_control]).output();
    let current = match output {
        Ok(o) => parse_volume(&String::from_utf8_lossy(&o.stdout)).unwrap_or(50),
        Err(_) => return,
    };
    let target = ((current as i32) + delta).clamp(0, config.max_volume as i32);
    let _ = std::process::Command::new("amixer")
        .args(["-c", &config.card_index.to_string(), "set", &config.mixer_control, &format!("{}%", target)]).output();
    info!("Volume adjusted to {}%", target);
}

fn start_key_listener(config: KeysConfig, audio: AudioConfig, led: LedHandle, state: Arc<KeyState>) {
    if !config.enabled { info!("Key listener disabled"); return; }
    std::thread::spawn(move || {
        use std::io::Read;
        let mut file = match std::fs::File::open(&config.device) {
            Ok(f) => f,
            Err(e) => { warn!("Cannot open {}: {}", config.device, e); return; }
        };
        info!("Key listener started on {}", config.device);
        let mut buf = [0u8; 16];
        loop {
            if file.read_exact(&mut buf).is_err() { std::thread::sleep(Duration::from_millis(100)); continue; }
            let ev_type = u16::from_le_bytes([buf[8], buf[9]]);
            let ev_code = u16::from_le_bytes([buf[10], buf[11]]);
            let ev_value = i32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
            if ev_type != 1 { continue; }
            if ev_code == config.mute && ev_value == 1 {
                let prev = state.muted.load(Ordering::SeqCst);
                let now = !prev;
                state.muted.store(now, Ordering::SeqCst);
                info!("Mute toggled: mic {}", if now { "OFF" } else { "ON" });
                led.send(LedCommand::SetState(if now { LedState::Muted } else { LedState::Idle }));
            } else if ev_value == 1 {
                if ev_code == config.volume_up { adjust_volume_relative(&audio, config.volume_step as i32); }
                else if ev_code == config.volume_down { adjust_volume_relative(&audio, -(config.volume_step as i32)); }
                else if ev_code == config.play_pause {
                    info!("Play/pause button pressed, stopping playback");
                    state.stop_playback.store(true, Ordering::SeqCst);
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

fn check_binary(name: &str) -> DiagnosticResult {
    match std::process::Command::new("which").arg(name).output() {
        Ok(out) if out.status.success() => {
            let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
            DiagnosticResult::Ok(format!("found at {}", path))
        }
        _ => DiagnosticResult::Fail(format!("{} not found", name)),
    }
}

fn check_audio_devices(config: &AudioConfig) -> Vec<(&'static str, DiagnosticResult)> {
    let mut results = Vec::new();
    let host = cpal::default_host();
    match host.default_input_device() {
        Some(dev) => match dev.name() {
            Ok(name) => match dev.default_input_config() {
                Ok(cfg) => results.push(("Input device", DiagnosticResult::Ok(format!("'{}' ({} Hz, {} ch)", name, cfg.sample_rate().0, cfg.channels())))),
                Err(e) => results.push(("Input device", DiagnosticResult::Warn(format!("'{}' query failed: {}", name, e)))),
            },
            Err(e) => results.push(("Input device", DiagnosticResult::Warn(format!("name unknown: {}", e)))),
        },
        None => results.push(("Input device", DiagnosticResult::Fail("No default input device".into()))),
    }
    match host.default_output_device() {
        Some(dev) => match dev.name() {
            Ok(name) => match dev.default_output_config() {
                Ok(cfg) => results.push(("Output device", DiagnosticResult::Ok(format!("'{}' ({} Hz, {} ch)", name, cfg.sample_rate().0, cfg.channels())))),
                Err(e) => results.push(("Output device", DiagnosticResult::Warn(format!("'{}' query failed: {}", name, e)))),
            },
            Err(e) => results.push(("Output device", DiagnosticResult::Warn(format!("name unknown: {}", e)))),
        },
        None => results.push(("Output device", DiagnosticResult::Fail("No default output device".into()))),
    }
    match std::process::Command::new("amixer")
        .args(["-c", &config.card_index.to_string(), "get", &config.mixer_control]).output() {
        Ok(out) if out.status.success() => {
            let vol = parse_volume(&String::from_utf8_lossy(&out.stdout)).unwrap_or(0);
            results.push(("Mixer control", DiagnosticResult::Ok(format!("'{}' card {} = {}%", config.mixer_control, config.card_index, vol))));
        }
        Ok(out) => results.push(("Mixer control", DiagnosticResult::Warn(format!("amixer: {}", String::from_utf8_lossy(&out.stderr).trim())))),
        Err(e) => results.push(("Mixer control", DiagnosticResult::Fail(format!("amixer not runnable: {}", e)))),
    }
    results
}

fn check_input_device(config: &KeysConfig) -> DiagnosticResult {
    if !config.enabled { return DiagnosticResult::Warn("disabled".into()); }
    if PathBuf::from(&config.device).exists() { DiagnosticResult::Ok(config.device.clone()) }
    else { DiagnosticResult::Fail(format!("{} not found", config.device)) }
}

fn check_led(config: &LedConfig) -> DiagnosticResult {
    if !config.enabled { return DiagnosticResult::Warn("disabled in config".into()); }
    if !PathBuf::from(&config.device).exists() { return DiagnosticResult::Fail(format!("{} not found", config.device)); }
    let test = [0u32; LED_COUNT];
    match write_leds(&config.device, &test) {
        Ok(_) => DiagnosticResult::Ok(format!("{} writable", config.device)),
        Err(e) => DiagnosticResult::Fail(format!("write failed: {}", e)),
    }
}

async fn check_region(client: &reqwest::Client) -> Vec<DiagnosticResult> {
    let mut results = Vec::new();
    match client.get(IP_API_URL).send().await {
        Ok(resp) if resp.status().is_success() => match resp.json::<IpApiResponse>().await {
            Ok(info) => {
                let loc = format!("{}, {} ({})", info.city.as_deref().unwrap_or("?"), info.region_name.as_deref().unwrap_or("?"), info.country.as_deref().unwrap_or("?"));
                results.push(DiagnosticResult::Ok(format!("IP {} - {}", info.query.as_deref().unwrap_or("?"), loc)));
                if let Some(tz) = &info.timezone { results.push(DiagnosticResult::Ok(format!("timezone {}", tz))); }
                let code = info.country_code.as_deref().unwrap_or("");
                if code == "CN" || code == "HK" { results.push(DiagnosticResult::Warn("Region may be blocked by Groq API".into())); }
                else { results.push(DiagnosticResult::Ok("Region appears allowed for Groq API".into())); }
            }
            Err(e) => results.push(DiagnosticResult::Warn(format!("parse error: {}", e))),
        },
        Ok(resp) => results.push(DiagnosticResult::Warn(format!("HTTP {}", resp.status()))),
        Err(e) => results.push(DiagnosticResult::Fail(format!("cannot reach: {}", e))),
    }
    results
}

async fn check_groq_api(client: &reqwest::Client, config: &GroqConfig) -> Vec<DiagnosticResult> {
    let mut results = Vec::new();
    if config.api_key.trim().is_empty() {
        results.push(DiagnosticResult::Warn("API key not set".into()));
        return results;
    }
    match client.get(MODELS_URL).bearer_auth(&config.api_key).send().await {
        Ok(r) => {
            let status = r.status().as_u16();
            if status != 200 {
                let body = r.text().await.unwrap_or_default();
                results.push(DiagnosticResult::Fail(format!("API key rejected (HTTP {}): {}", status, body.chars().take(120).collect::<String>())));
                return results;
            }
            results.push(DiagnosticResult::Ok("API key accepted".into()));
            match r.json::<ModelsResponse>().await {
                Ok(models) => {
                    let ids: Vec<String> = models.data.iter().map(|m| m.id.clone()).collect();
                    for (label, id) in [("STT model", &config.stt_model), ("LLM model", &config.llm_model), ("TTS model", &config.tts_model)] {
                        if ids.iter().any(|i| i == id) { results.push(DiagnosticResult::Ok(format!("{} available", label))); }
                        else { results.push(DiagnosticResult::Warn(format!("{} '{}' not in account", label, id))); }
                    }
                }
                Err(e) => results.push(DiagnosticResult::Warn(format!("cannot parse models: {}", e))),
            }
        }
        Err(e) => results.push(DiagnosticResult::Fail(format!("cannot reach Groq: {}", e))),
    }
    results
}

fn count_result(r: &DiagnosticResult, fails: &mut usize, warns: &mut usize) {
    match r { DiagnosticResult::Fail(_) => *fails += 1, DiagnosticResult::Warn(_) => *warns += 1, DiagnosticResult::Ok(_) => {} }
}

async fn run_diagnostics(config: &Config, client: &reqwest::Client) -> bool {
    println!();
    println!("============================================================");
    println!("  Rimth Device Diagnostics");
    println!("============================================================");
    let mut fails = 0usize; let mut warns = 0usize;

    println!(); println!("[System]");
    let system_items: Vec<(&str, DiagnosticResult)> = vec![
        ("Config file", match config_path() {
            Ok(p) if p.exists() => DiagnosticResult::Ok(format!("{}", p.display())),
            Ok(p) => DiagnosticResult::Warn(format!("missing at {}", p.display())),
            Err(e) => DiagnosticResult::Fail(format!("cannot resolve: {}", e)),
        }),
        ("amixer", check_binary("amixer")),
        ("sh", check_binary("sh")),
        ("date", check_binary("date")),
        ("sounds dir", match sounds_dir() {
            Ok(p) if p.exists() => DiagnosticResult::Ok(format!("{}", p.display())),
            Ok(p) => DiagnosticResult::Warn(format!("missing at {}", p.display())),
            Err(e) => DiagnosticResult::Fail(format!("cannot resolve: {}", e)),
        }),
    ];
    for (label, r) in &system_items { r.print(label); count_result(r, &mut fails, &mut warns); }

    println!(); println!("[Audio]");
    for (label, r) in check_audio_devices(&config.audio) { r.print(label); count_result(&r, &mut fails, &mut warns); }

    println!(); println!("[Input]");
    let input_r = check_input_device(&config.keys);
    input_r.print("Input device");
    count_result(&input_r, &mut fails, &mut warns);

    println!(); println!("[LED]");
    let led_r = check_led(&config.led);
    led_r.print("LED device");
    count_result(&led_r, &mut fails, &mut warns);

    println!(); println!("[Network]");
    if config.proxy.url.is_empty() {
        let r = DiagnosticResult::Warn("proxy not configured".into());
        r.print("Proxy"); count_result(&r, &mut fails, &mut warns);
    } else {
        let r = DiagnosticResult::Ok(config.proxy.url.clone());
        r.print("Proxy"); count_result(&r, &mut fails, &mut warns);
    }
    for r in check_region(client).await { r.print("Region"); count_result(&r, &mut fails, &mut warns); }

    println!(); println!("[Groq API]");
    for r in check_groq_api(client, &config.groq).await { r.print("Groq"); count_result(&r, &mut fails, &mut warns); }

    println!();
    println!("============================================================");
    println!("  Diagnostics complete: {} failure(s), {} warning(s)", fails, warns);
    println!("============================================================");
    println!();
    fails == 0
}

// ---------------------------------------------------------------------------
// Audio capture
// ---------------------------------------------------------------------------

fn start_capture(tx: Sender<Vec<f32>>, sample_rate: u32, mic_gain: Arc<AtomicU32>) -> Result<Stream> {
    let host = cpal::default_host();
    let device = host.default_input_device().context("No input device found")?;
    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };
    let fmt = device.default_input_config()?.sample_format();
    let stream = match fmt {
        SampleFormat::F32 => device.build_input_stream(
            &config,
            move |d: &[f32], _| {
                let gain = f32::from_bits(mic_gain.load(Ordering::Relaxed));
                let v: Vec<f32> = d.iter().map(|&s| s * gain).collect();
                let _ = tx.try_send(v);
            },
            |e| eprintln!("stream err: {}", e), None)?,
        SampleFormat::I16 => {
            let tx = tx.clone();
            device.build_input_stream(
                &config,
                move |d: &[i16], _| {
                    let gain = f32::from_bits(mic_gain.load(Ordering::Relaxed));
                    let v: Vec<f32> = d.iter().map(|&s| s as f32 / 32768.0 * gain).collect();
                    let _ = tx.try_send(v);
                },
                |e| eprintln!("stream err: {}", e), None)?
        }
        f => return Err(anyhow::anyhow!("Unsupported format: {:?}", f)),
    };
    stream.play()?;
    Ok(stream)
}

async fn wait_trigger(
    rx: &mut Receiver<Vec<f32>>,
    config: &AudioConfig,
    is_playing: &Arc<AtomicBool>,
    key_state: &KeyState,
    capture: &CaptureCtx,
) -> Option<Vec<f32>> {
    loop {
        match tokio::time::timeout(Duration::from_secs(CAPTURE_IDLE_TIMEOUT_SECS), rx.recv()).await {
            Ok(Some(chunk)) => {
                if is_playing.load(Ordering::Relaxed) { continue; }
                if key_state.muted.load(Ordering::Relaxed) { continue; }
                if key_state.conversation_mode.load(Ordering::Relaxed) { return Some(Vec::new()); }
                if rms(&chunk) > config.trigger_threshold {
                    info!("Volume trigger detected");
                    return Some(chunk);
                }
            }
            Ok(None) => return None,
            Err(_) => {
                warn!("No audio data for {}s, attempting capture restart", CAPTURE_IDLE_TIMEOUT_SECS);
                if let Err(e) = capture.restart() {
                    warn!("Capture restart failed: {}", e);
                }
            }
        }
    }
}

async fn record(
    rx: &mut Receiver<Vec<f32>>,
    config: &AudioConfig,
    is_playing: &Arc<AtomicBool>,
    key_state: &KeyState,
    initial: Vec<f32>,
) -> Result<Option<Vec<u8>>> {
    let mut samples: Vec<f32> = initial;
    let mut silence: usize = 0;
    let silence_limit = config.sample_rate as usize * config.silence_duration_ms as usize / 1000;
    let min_samples = config.sample_rate as usize * config.min_recording_ms as usize / 1000;
    let max_samples = config.sample_rate as usize * 30;

    while let Some(chunk) = rx.recv().await {
        if is_playing.load(Ordering::Relaxed) { continue; }
        if key_state.muted.load(Ordering::Relaxed) {
            info!("Recording cancelled: muted");
            return Ok(None);
        }
        let level = rms(&chunk);
        samples.extend_from_slice(&chunk);
        if level < config.silence_threshold { silence += chunk.len(); } else { silence = 0; }
        if samples.len() >= min_samples && silence >= silence_limit { break; }
        if samples.len() > max_samples { break; }
    }

    let spec = WavSpec { channels: 1, sample_rate: config.sample_rate, bits_per_sample: 16, sample_format: HoundSampleFormat::Int };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut w = WavWriter::new(&mut cursor, spec)?;
        for &s in &samples { w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?; }
        w.finalize()?;
    }
    Ok(Some(cursor.into_inner()))
}

// ---------------------------------------------------------------------------
// Audio playback (cpal based, shared device friendly)
// ---------------------------------------------------------------------------

fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32, u16)> {
    let reader = WavReader::new(Cursor::new(bytes)).context("Invalid WAV")?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        HoundSampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
        HoundSampleFormat::Float => reader.into_samples::<f32>().filter_map(|s| s.ok()).collect(),
    };
    Ok((samples, spec.sample_rate, spec.channels))
}

fn play_audio_bytes(bytes: Vec<u8>, stop_flag: &Arc<AtomicBool>) -> Result<()> {
    if stop_flag.load(Ordering::SeqCst) {
        stop_flag.store(false, Ordering::SeqCst);
        return Ok(());
    }

    let _guard = OUTPUT_LOCK.lock().unwrap();

    let (raw, in_rate, in_ch) = decode_wav(&bytes)?;
    if raw.is_empty() { return Ok(()); }

    let in_ch = in_ch as usize;

    // Downmix to mono
    let mono: Vec<f32> = if in_ch <= 1 {
        raw
    } else {
        raw.chunks(in_ch).map(|c| c.iter().sum::<f32>() / in_ch as f32).collect()
    };

    let host = cpal::default_host();
    let device = host.default_output_device().context("No default output device")?;
    let default_cfg = device.default_output_config()?;
    let out_rate = default_cfg.sample_rate().0;
    let out_ch = default_cfg.channels() as usize;
    let out_fmt = default_cfg.sample_format();

    // Simple linear resample if needed
    let resampled: Vec<f32> = if in_rate == out_rate {
        mono
    } else {
        let ratio = out_rate as f64 / in_rate as f64;
        let new_len = ((mono.len() as f64) * ratio) as usize;
        let mut out = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src = i as f64 / ratio;
            let i0 = src.floor() as usize;
            let i1 = (i0 + 1).min(mono.len().saturating_sub(1));
            let t = (src - i0 as f64) as f32;
            let s0 = mono.get(i0).copied().unwrap_or(0.0);
            let s1 = mono.get(i1).copied().unwrap_or(0.0);
            out.push(s0 * (1.0 - t) + s1 * t);
        }
        out
    };

    let interleaved: Vec<f32> = if out_ch <= 1 {
        resampled
    } else {
        let mut v = Vec::with_capacity(resampled.len() * out_ch);
        for s in resampled {
            for _ in 0..out_ch { v.push(s); }
        }
        v
    };

    let samples_arc = Arc::new(interleaved);
    let pos = Arc::new(AtomicUsize::new(0));
    let finished = Arc::new(AtomicBool::new(false));

    let stream_config = cpal::StreamConfig {
        channels: out_ch as u16,
        sample_rate: cpal::SampleRate(out_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let s_cb = samples_arc.clone();
    let p_cb = pos.clone();
    let f_cb = finished.clone();

    let err_fn = |e| eprintln!("output stream err: {}", e);

    let stream = match out_fmt {
        SampleFormat::F32 => device.build_output_stream(
            &stream_config,
            move |out: &mut [f32], _| {
                let p = p_cb.load(Ordering::Relaxed);
                let rem = s_cb.len().saturating_sub(p);
                let n = out.len().min(rem);
                out[..n].copy_from_slice(&s_cb[p..p + n]);
                for s in &mut out[n..] { *s = 0.0; }
                p_cb.store(p + n, Ordering::Relaxed);
                if n < out.len() { f_cb.store(true, Ordering::Relaxed); }
            },
            err_fn, None,
        )?,
        SampleFormat::I16 => {
            let i16_samples: Arc<Vec<i16>> = Arc::new(
                s_cb.iter().map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16).collect()
            );
            let s_cb2 = i16_samples.clone();
            device.build_output_stream(
                &stream_config,
                move |out: &mut [i16], _| {
                    let p = p_cb.load(Ordering::Relaxed);
                    let rem = s_cb2.len().saturating_sub(p);
                    let n = out.len().min(rem);
                    out[..n].copy_from_slice(&s_cb2[p..p + n]);
                    for s in &mut out[n..] { *s = 0; }
                    p_cb.store(p + n, Ordering::Relaxed);
                    if n < out.len() { f_cb.store(true, Ordering::Relaxed); }
                },
                err_fn, None,
            )?
        }
        f => return Err(anyhow::anyhow!("Unsupported output format: {:?}", f)),
    };

    stream.play()?;

    while !finished.load(Ordering::Relaxed) {
        if stop_flag.load(Ordering::SeqCst) {
            stop_flag.store(false, Ordering::SeqCst);
            info!("Playback stopped by key");
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    drop(stream);
    // Small tail so the hardware finishes the last buffer
    std::thread::sleep(Duration::from_millis(60));
    Ok(())
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

fn prompt_path(p: Prompt) -> PathBuf {
    let name = match p {
        Prompt::Auth => "auth_error.wav", Prompt::RateLimit => "rate_limited.wav",
        Prompt::BadRequest => "bad_request.wav", Prompt::Server => "server_error.wav",
        Prompt::Network => "network_error.wav",
    };
    sounds_dir().map(|d| d.join(name)).unwrap_or_else(|_| PathBuf::from(name))
}

async fn generate_prompt_sounds(client: &reqwest::Client, config: &GroqConfig) -> Result<()> {
    if config.api_key.trim().is_empty() { return Ok(()); }
    let dir = sounds_dir()?;
    std::fs::create_dir_all(&dir)?;
    let prompts: Vec<(Prompt, &str)> = vec![
        (Prompt::Auth, "Authentication error. Please check the API key."),
        (Prompt::RateLimit, "Rate limit exceeded. Please wait a moment."),
        (Prompt::BadRequest, "Bad request. The input was not valid."),
        (Prompt::Server, "Server error. Please try again later."),
        (Prompt::Network, "Network error. Please check the connection."),
    ];
    for (p, text) in prompts {
        let path = prompt_path(p);
        if path.exists() { continue; }
        info!("Generating prompt sound: {}", path.display());
        match tts(client, config, text, config.tts_voice.as_str()).await {
            Ok(audio) => { std::fs::write(&path, audio)?; }
            Err(e) => warn!("Failed to generate prompt: {}", e),
        }
    }
    Ok(())
}

async fn play_prompt(p: Prompt, stop_flag: &Arc<AtomicBool>) {
    let path = prompt_path(p);
    if !path.exists() { warn!("Missing prompt sound: {}", path.display()); return; }
    let stop = stop_flag.clone();
    let _ = tokio::task::spawn_blocking(move || {
        let bytes = match std::fs::read(&path) { Ok(b) => b, Err(e) => { warn!("Read prompt failed: {}", e); return; } };
        if let Err(e) = play_audio_bytes(bytes, &stop) { warn!("Play prompt failed: {}", e); }
    }).await;
}

// ---------------------------------------------------------------------------
// Groq API
// ---------------------------------------------------------------------------

async fn stt(client: &reqwest::Client, config: &GroqConfig, wav: Vec<u8>) -> Result<String, SttError> {
    let part = multipart::Part::bytes(wav).file_name("audio.wav").mime_str("audio/wav")
        .map_err(|e| SttError { status: 0, message: e.to_string() })?;
    let form = multipart::Form::new()
        .part("file", part)
        .text("model", config.stt_model.clone())
        .text("language", config.language.clone())
        .text("temperature", "0.0");
    let resp = client.post(STT_URL).bearer_auth(&config.api_key).multipart(form).send().await
        .map_err(|e| SttError { status: 0, message: e.to_string() })?;
    let status = resp.status().as_u16();
    if status == 200 {
        let body: SttResponse = resp.json().await.map_err(|e| SttError { status, message: e.to_string() })?;
        Ok(body.text)
    } else {
        let msg = resp.text().await.unwrap_or_default();
        Err(SttError { status, message: msg })
    }
}

async fn tts(client: &reqwest::Client, config: &GroqConfig, text: &str, voice: &str) -> Result<Vec<u8>> {
    let body = serde_json::json!({
        "model": config.tts_model,
        "input": text,
        "voice": voice,
        "response_format": "wav"
    });
    let resp = client.post(TTS_URL).bearer_auth(&config.api_key).json(&body).send().await?;
    let status = resp.status().as_u16();
    if status != 200 {
        let msg = resp.text().await.unwrap_or_default();
        anyhow::bail!("TTS {}: {}", status, msg);
    }
    Ok(resp.bytes().await?.to_vec())
}

async fn chat(client: &reqwest::Client, config: &GroqConfig, messages: &[ChatMessage], tools: Option<serde_json::Value>) -> Result<ChatMessageResponse> {
    let mut body = serde_json::json!({
        "model": config.llm_model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 256
    });
    if let Some(t) = tools { body["tools"] = t; }
    let resp = client.post(CHAT_URL).bearer_auth(&config.api_key).json(&body).send().await?;
    let status = resp.status().as_u16();
    if status != 200 {
        let msg = resp.text().await.unwrap_or_default();
        anyhow::bail!("Chat {}: {}", status, msg);
    }
    let body: ChatResponse = resp.json().await?;
    body.choices.into_iter().next().map(|c| c.message).ok_or_else(|| anyhow::anyhow!("No choices"))
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

fn tools_definition() -> serde_json::Value {
    serde_json::json!([
        {"type": "function", "function": {"name": "set_volume",
            "description": "Set the speaker volume to an absolute percentage (0-100).",
            "parameters": {"type": "object", "properties": {"percent": {"type": "integer"}}, "required": ["percent"]}}},
        {"type": "function", "function": {"name": "get_volume",
            "description": "Get the current speaker volume percentage.",
            "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "adjust_volume",
            "description": "Adjust the speaker volume by a relative delta.",
            "parameters": {"type": "object", "properties": {"delta": {"type": "integer"}}, "required": ["delta"]}}},
        {"type": "function", "function": {"name": "run_terminal",
            "description": "Run a shell command on the speaker.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
        {"type": "function", "function": {"name": "fetch_url",
            "description": "Fetch the content of a URL.",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
        {"type": "function", "function": {"name": "run_diagnostics",
            "description": "Run device diagnostics.",
            "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "switch_model",
            "description": "Switch the STT, LLM, TTS model or TTS voice.",
            "parameters": {"type": "object", "properties": {
                "model_type": {"type": "string", "enum": ["stt", "llm", "tts", "voice"]},
                "model_name": {"type": "string"}}, "required": ["model_type", "model_name"]}}},
        {"type": "function", "function": {"name": "set_alarm",
            "description": "Set an alarm on the speaker.",
            "parameters": {"type": "object", "properties": {
                "time": {"type": "string"}, "label": {"type": "string"},
                "repeat": {"type": "string", "enum": ["once", "daily", "weekdays"]}}, "required": ["time"]}}},
        {"type": "function", "function": {"name": "list_alarms",
            "description": "List all set alarms.",
            "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "delete_alarm",
            "description": "Delete an alarm by its ID.",
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}},
        {"type": "function", "function": {"name": "led_control",
            "description": "Control the RGB LED ring on the speaker.",
            "parameters": {"type": "object", "properties": {
                "color": {"type": "string"}, "index": {"type": "integer"},
                "brightness": {"type": "integer"},
                "state": {"type": "string", "enum": ["idle", "listening", "speaking", "muted", "off"]}}}}},
        {"type": "function", "function": {"name": "conversation_mode",
            "description": "Enable or disable conversation mode.",
            "parameters": {"type": "object", "properties": {"enabled": {"type": "boolean"}}, "required": ["enabled"]}}},
        {"type": "function", "function": {"name": "session_control",
            "description": "Manage sessions.",
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["new", "list", "switch"]},
                "session_name": {"type": "string"}}, "required": ["action"]}}},
        {"type": "function", "function": {"name": "set_mic_gain",
            "description": "Set the microphone gain multiplier (0.1 - 10.0).",
            "parameters": {"type": "object", "properties": {"gain": {"type": "number"}}, "required": ["gain"]}}},
        {"type": "function", "function": {"name": "get_mic_gain",
            "description": "Get the current microphone gain.",
            "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "set_mute",
            "description": "Toggle or set the microphone mute state.",
            "parameters": {"type": "object", "properties": {"muted": {"type": "boolean"}}, "required": ["muted"]}}},
        {"type": "function", "function": {"name": "query_history",
            "description": "Return the most recent turns of the current conversation. Use this when the user asks what was said earlier or wants a recap.",
            "parameters": {"type": "object", "properties": {"count": {"type": "integer"}}, "required": []}}},
        {"type": "function", "function": {"name": "clear_history",
            "description": "Forget the current conversation, keeping only the system prompt.",
            "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "get_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "get_system_info",
            "description": "Get basic device information: uptime, load average and memory usage.",
            "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "stop_speaking",
            "description": "Stop the current speech playback immediately.",
            "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "set_language",
            "description": "Change the STT and reply language code (e.g. en, zh, ja, ko, es).",
            "parameters": {"type": "object", "properties": {"language": {"type": "string"}}, "required": ["language"]}}}
    ])
}

async fn amixer_get(config: &AudioConfig) -> Result<u32> {
    let output = tokio::process::Command::new("amixer")
        .args(["-c", &config.card_index.to_string(), "get", &config.mixer_control]).output().await?;
    parse_volume(&String::from_utf8_lossy(&output.stdout)).ok_or_else(|| anyhow::anyhow!("Cannot parse volume"))
}

async fn amixer_set(config: &AudioConfig, percent: u32) -> Result<u32> {
    let clamped = percent.min(config.max_volume);
    let output = tokio::process::Command::new("amixer")
        .args(["-c", &config.card_index.to_string(), "set", &config.mixer_control, &format!("{}%", clamped)]).output().await?;
    if !output.status.success() { anyhow::bail!("amixer failed"); }
    Ok(clamped)
}

async fn execute_tool(
    client: &reqwest::Client,
    audio_config: &AudioConfig,
    alarm_config: &AlarmConfig,
    config: &Config,
    state: &Arc<AppState>,
    name: &str,
    args: &str,
) -> String {
    match name {
        "set_volume" => {
            let parsed: serde_json::Value = match serde_json::from_str(args) { Ok(v) => v, Err(e) => return format!("Parse failed: {}", e) };
            let percent = parsed["percent"].as_u64().unwrap_or(50) as u32;
            match amixer_set(audio_config, percent).await { Ok(a) => format!("Volume set to {}%", a), Err(e) => format!("Failed: {}", e) }
        }
        "get_volume" => match amixer_get(audio_config).await { Ok(v) => format!("Current volume is {}%", v), Err(e) => format!("Failed: {}", e) },
        "adjust_volume" => {
            let parsed: serde_json::Value = match serde_json::from_str(args) { Ok(v) => v, Err(e) => return format!("Parse failed: {}", e) };
            let delta = parsed["delta"].as_i64().unwrap_or(0);
            match amixer_get(audio_config).await {
                Ok(current) => {
                    let target = (current as i64 + delta).clamp(0, audio_config.max_volume as i64);
                    match amixer_set(audio_config, target as u32).await { Ok(a) => format!("Volume adjusted to {}%", a), Err(e) => format!("Failed: {}", e) }
                }
                Err(e) => format!("Failed: {}", e),
            }
        }
        "run_terminal" => {
            let parsed: serde_json::Value = match serde_json::from_str(args) { Ok(v) => v, Err(e) => return format!("Parse failed: {}", e) };
            let command = parsed["command"].as_str().unwrap_or("");
            if command.is_empty() { return "No command".into(); }
            info!("Running: {}", command);
            match tokio::process::Command::new("sh").arg("-c").arg(command).output().await {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let mut result = String::new();
                    if !stdout.is_empty() { result.push_str(&stdout); }
                    if !stderr.is_empty() { if !result.is_empty() { result.push('\n'); } result.push_str("STDERR: "); result.push_str(&stderr); }
                    if result.len() > 2000 { result.truncate(2000); result.push_str("... (truncated)"); }
                    result
                }
                Err(e) => format!("Failed: {}", e),
            }
        }
        "fetch_url" => {
            let parsed: serde_json::Value = match serde_json::from_str(args) { Ok(v) => v, Err(e) => return format!("Parse failed: {}", e) };
            let url = parsed["url"].as_str().unwrap_or("");
            if url.is_empty() { return "No URL".into(); }
            info!("Fetching: {}", url);
            match client.get(url).send().await {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let text = resp.text().await.unwrap_or_default();
                    let mut result = format!("Status: {}\n", status);
                    if text.len() > 2000 { result.push_str(&text[..2000]); result.push_str("... (truncated)"); }
                    else { result.push_str(&text); }
                    result
                }
                Err(e) => format!("Failed: {}", e),
            }
        }
        "run_diagnostics" => {
            let ok = run_diagnostics(config, client).await;
            if ok { "Diagnostics finished with no failures".into() }
            else { "Diagnostics finished with failures, check console".into() }
        }
        "switch_model" => {
            let parsed: serde_json::Value = match serde_json::from_str(args) { Ok(v) => v, Err(e) => return format!("Parse failed: {}", e) };
            let model_type = parsed["model_type"].as_str().unwrap_or("");
            let model_name = parsed["model_name"].as_str().unwrap_or("");
            if model_name.is_empty() { return "No model name".into(); }
            let mut cfg = state.groq.write().await;
            match model_type {
                "stt" => { cfg.stt_model = model_name.into(); }
                "llm" => { cfg.llm_model = model_name.into(); }
                "tts" => { cfg.tts_model = model_name.into(); }
                "voice" => { cfg.tts_voice = model_name.into(); }
                _ => return format!("Unknown model_type: {}", model_type),
            }
            info!("Switched {} to {}", model_type, model_name);
            format!("Switched {} to {}", model_type, model_name)
        }
        "set_alarm" => {
            let parsed: serde_json::Value = match serde_json::from_str(args) { Ok(v) => v, Err(e) => return format!("Parse failed: {}", e) };
            let time = parsed["time"].as_str().unwrap_or("07:00");
            let label = parsed["label"].as_str().unwrap_or("Alarm");
            let repeat = parsed["repeat"].as_str().unwrap_or("once");
            add_alarm(&alarm_config.store_path, time, label, repeat)
        }
        "list_alarms" => list_alarms_text(&alarm_config.store_path),
        "delete_alarm" => {
            let parsed: serde_json::Value = match serde_json::from_str(args) { Ok(v) => v, Err(e) => return format!("Parse failed: {}", e) };
            let id = parsed["id"].as_str().unwrap_or("");
            delete_alarm(&alarm_config.store_path, id)
        }
        "led_control" => {
            let parsed: serde_json::Value = match serde_json::from_str(args) { Ok(v) => v, Err(e) => return format!("Parse failed: {}", e) };
            let mut actions: Vec<String> = Vec::new();
            if let Some(state_str) = parsed["state"].as_str() {
                let led_state = match state_str {
                    "idle" => Some(LedState::Idle), "listening" => Some(LedState::Listening),
                    "speaking" => Some(LedState::Speaking), "muted" => Some(LedState::Muted),
                    "off" => { state.led.send(LedCommand::Off); actions.push("LED off".into()); None }
                    _ => None,
                };
                if let Some(s) = led_state { state.led.send(LedCommand::SetState(s)); actions.push(format!("state {}", state_str)); }
            }
            if let Some(color) = parsed["color"].as_str() {
                if parse_rgb_hex_to_bgr(color).is_some() {
                    if let Some(idx) = parsed["index"].as_i64() {
                        if (0..LED_COUNT as i64).contains(&idx) {
                            state.led.send(LedCommand::SetOneHex(idx as usize, color.to_string()));
                            actions.push(format!("LED {} = #{}", idx, color));
                        } else {
                            state.led.send(LedCommand::SetColorHex(color.to_string()));
                            actions.push(format!("all LEDs = #{}", color));
                        }
                    } else {
                        state.led.send(LedCommand::SetColorHex(color.to_string()));
                        actions.push(format!("all LEDs = #{}", color));
                    }
                }
            }
            if let Some(b) = parsed["brightness"].as_u64() {
                let b = b.min(100) as u8;
                state.led.send(LedCommand::SetBrightness(b));
                actions.push(format!("brightness {}", b));
            }
            if actions.is_empty() { "No LED action specified".into() } else { format!("LED: {}", actions.join(", ")) }
        }
        "conversation_mode" => {
            let parsed: serde_json::Value = match serde_json::from_str(args) { Ok(v) => v, Err(e) => return format!("Parse failed: {}", e) };
            let enabled = parsed["enabled"].as_bool().unwrap_or(false);
            state.key_state.conversation_mode.store(enabled, Ordering::SeqCst);
            if enabled { "Conversation mode enabled. I will keep listening.".into() } else { "Conversation mode disabled.".into() }
        }
        "session_control" => {
            let parsed: serde_json::Value = match serde_json::from_str(args) { Ok(v) => v, Err(e) => return format!("Parse failed: {}", e) };
            let action = parsed["action"].as_str().unwrap_or("list");
            match action {
                "new" => {
                    let requested = parsed["session_name"].as_str().unwrap_or("").trim();
                    let new_name = if requested.is_empty() {
                        format!("session_{}", rand::random::<u16>())
                    } else {
                        requested.to_string()
                    };
                    {
                        let mut cur = state.current_session.lock().unwrap();
                        // Persist current session before switching
                        let msgs = state.messages.lock().unwrap().clone();
                        save_session(&state.sessions_dir, &cur, &msgs);
                        *cur = new_name.clone();
                    }
                    {
                        let mut msgs = state.messages.lock().unwrap();
                        msgs.clear();
                        msgs.push(ChatMessage::system(state.effective_system.lock().unwrap().clone()));
                    }
                    info!("Started new session: {}", new_name);
                    format!("New session started: {}", new_name)
                }
                "list" => {
                    let sessions = list_sessions(&state.sessions_dir);
                    if sessions.is_empty() { "No sessions".into() } else { format!("Sessions: {}", sessions.join(", ")) }
                }
                "switch" => {
                    let name = parsed["session_name"].as_str().unwrap_or("").trim().to_string();
                    if name.is_empty() { return "No session name provided".into(); }
                    let sessions = list_sessions(&state.sessions_dir);
                    if !sessions.contains(&name) {
                        return format!("Session '{}' not found. Available: {}", name, sessions.join(", "));
                    }
                    // Save current session before switching
                    {
                        let mut cur = state.current_session.lock().unwrap();
                        let msgs = state.messages.lock().unwrap().clone();
                        save_session(&state.sessions_dir, &cur, &msgs);
                        *cur = name.clone();
                    }
                    let mut loaded = load_session(&state.sessions_dir, &name);
                    if loaded.is_empty() || loaded[0].role != "system" {
                        loaded.insert(0, ChatMessage::system(state.effective_system.lock().unwrap().clone()));
                    }
                    {
                        let mut msgs = state.messages.lock().unwrap();
                        *msgs = loaded;
                    }
                    info!("Switched to session: {}", name);
                    format!("Switched to session: {}", name)
                }
                _ => "Unknown session action".into(),
            }
        }
        "set_mic_gain" => {
            let parsed: serde_json::Value = match serde_json::from_str(args) { Ok(v) => v, Err(e) => return format!("Parse failed: {}", e) };
            let gain = parsed["gain"].as_f64().unwrap_or(1.0);
            let clamped = (gain as f32).clamp(0.1, 10.0);
            state.mic_gain.store(clamped.to_bits(), Ordering::SeqCst);
            format!("Microphone gain set to {:.1}", clamped)
        }
        "get_mic_gain" => {
            let gain = f32::from_bits(state.mic_gain.load(Ordering::Relaxed));
            format!("Microphone gain is {:.1}", gain)
        }
        "set_mute" => {
            let parsed: serde_json::Value = match serde_json::from_str(args) { Ok(v) => v, Err(e) => return format!("Parse failed: {}", e) };
            let muted = parsed["muted"].as_bool().unwrap_or(false);
            state.key_state.muted.store(muted, Ordering::SeqCst);
            if muted { state.led.send(LedCommand::SetState(LedState::Muted)); "Microphone muted".into() }
            else { state.led.send(LedCommand::SetState(LedState::Idle)); "Microphone unmuted".into() }
        }
        "query_history" => {
            let parsed: serde_json::Value = serde_json::from_str(args).unwrap_or(serde_json::json!({}));
            let count = parsed["count"].as_u64().unwrap_or(8).clamp(1, 30) as usize;
            let msgs = state.messages.lock().unwrap().clone();
            let recent: Vec<String> = msgs.iter()
                .filter(|m| m.role == "user" || m.role == "assistant")
                .filter_map(|m| {
                    let content = m.content.as_deref().unwrap_or("").trim();
                    if content.is_empty() { None }
                    else { Some(format!("{}: {}", m.role, content)) }
                })
                .collect();
            if recent.is_empty() { "No conversation history yet.".into() }
            else {
                let start = recent.len().saturating_sub(count);
                recent[start..].join("\n")
            }
        }
        "clear_history" => {
            {
                let mut msgs = state.messages.lock().unwrap();
                msgs.clear();
                msgs.push(ChatMessage::system(state.effective_system.lock().unwrap().clone()));
            }
            let cur = state.current_session.lock().unwrap().clone();
            let msgs = state.messages.lock().unwrap().clone();
            save_session(&state.sessions_dir, &cur, &msgs);
            "Conversation history cleared.".into()
        }
        "get_time" => local_time_hms(),
        "get_system_info" => {
            let uptime = std::fs::read_to_string("/proc/uptime")
                .ok()
                .and_then(|s| s.split_whitespace().next().map(|x| x.to_string()))
                .map(|s| format!("{:.0}s", s.parse::<f64>().unwrap_or(0.0)))
                .unwrap_or_else(|| "unknown".into());
            let loadavg = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
            let load = loadavg.split_whitespace().take(3).collect::<Vec<_>>().join(" ");
            let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
            let total_kb = meminfo.lines().find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1)).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
            let avail_kb = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1)).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
            format!("Uptime {}s, load {}, memory {}/{} MB used",
                    uptime, load, (total_kb - avail_kb) / 1024, total_kb / 1024)
        }
        "stop_speaking" => {
            state.key_state.stop_playback.store(true, Ordering::SeqCst);
            "Stopping playback".into()
        }
        "set_language" => {
            let parsed: serde_json::Value = match serde_json::from_str(args) { Ok(v) => v, Err(e) => return format!("Parse failed: {}", e) };
            let lang = parsed["language"].as_str().unwrap_or("en").trim().to_lowercase();
            if lang.is_empty() { return "Empty language code".into(); }
            {
                let mut cfg = state.groq.write().await;
                cfg.language = lang.clone();
            }
            info!("Language switched to {}", lang);
            format!("Language set to {}", lang)
        }
        _ => format!("Unknown tool: {}", name),
    }
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

fn session_file_path(dir: &PathBuf, name: &str) -> PathBuf { dir.join(format!("{}.json", name)) }

fn sanitize_session(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    let mut out: Vec<ChatMessage> = Vec::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => { if let Some(c) = msg.content { out.push(ChatMessage::system(c)); } }
            "user" => { if let Some(c) = msg.content { out.push(ChatMessage::user(c)); } }
            "assistant" => {
                if msg.tool_calls.is_none() {
                    if let Some(c) = msg.content {
                        if !c.trim().is_empty() { out.push(ChatMessage::assistant(Some(c), None)); }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

fn load_session(dir: &PathBuf, name: &str) -> Vec<ChatMessage> {
    let path = session_file_path(dir, name);
    if path.exists() {
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(messages) = serde_json::from_str::<Vec<ChatMessage>>(&content) {
                let cleaned = sanitize_session(messages);
                info!("Loaded session '{}' with {} messages (after sanitize)", name, cleaned.len());
                return cleaned;
            }
        }
    }
    Vec::new()
}

fn save_session(dir: &PathBuf, name: &str, messages: &[ChatMessage]) {
    let path = session_file_path(dir, name);
    if let Ok(content) = serde_json::to_string_pretty(messages) { let _ = std::fs::write(&path, content); }
}

fn list_sessions(dir: &PathBuf) -> Vec<String> {
    let mut names = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".json") { names.push(name.trim_end_matches(".json").to_string()); }
            }
        }
    }
    names.sort();
    names
}

/// Trim history without breaking tool_call / tool pairing.
fn trim_history(messages: &mut Vec<ChatMessage>, max: usize) {
    if messages.len() <= max + 1 { return; }
    let mut remove_count = messages.len() - max - 1;
    while remove_count > 0 && messages.len() > 2 {
        let can_remove = {
            let m = &messages[1];
            if m.role == "tool" { false }
            else if m.role == "assistant" && m.tool_calls.is_some() { false }
            else { true }
        };
        if !can_remove { break; }
        messages.remove(1);
        remove_count -= 1;
    }
}

// ---------------------------------------------------------------------------
// Speak
// ---------------------------------------------------------------------------

async fn speak(
    client: &reqwest::Client,
    config: &GroqConfig,
    text: &str,
    is_playing: &Arc<AtomicBool>,
    stop_playback: &Arc<AtomicBool>,
    rx: &mut Receiver<Vec<f32>>,
    led: &LedHandle,
) {
    let cleaned = strip_markdown(text);
    if cleaned.trim().is_empty() {
        warn!("Empty text after markdown strip, skipping TTS");
        return;
    }

    let lang = detect_language(&cleaned);
    let (_tts_model, tts_voice) = choose_tts_for_language(lang, config);
    info!("TTS language={}, voice={}", lang, tts_voice);

    let segments = split_text_for_tts(&cleaned, TTS_CHUNK_CHARS);
    info!("TTS: {} segment(s)", segments.len());

    is_playing.store(true, Ordering::SeqCst);
    while rx.try_recv().is_ok() {}
    led.send(LedCommand::SetState(LedState::Speaking));

    let cfg_snapshot = config.clone();
    let mut handles: VecDeque<tokio::task::JoinHandle<Result<Vec<u8>>>> = VecDeque::new();

    // Prime the pipeline
    let prefetch = TTS_PREFETCH.min(segments.len());
    for i in 0..prefetch {
        let seg = segments[i].clone();
        let cl = client.clone();
        let cfg = cfg_snapshot.clone();
        let v = tts_voice.clone();
        handles.push_back(tokio::spawn(async move { tts(&cl, &cfg, &seg, &v).await }));
    }

    for i in 0..segments.len() {
        if stop_playback.load(Ordering::SeqCst) {
            stop_playback.store(false, Ordering::SeqCst);
            info!("Playback cancelled before segment {}", i + 1);
            break;
        }

        // Prefetch the next unseen segment
        let next_i = i + TTS_PREFETCH;
        if next_i < segments.len() {
            let seg = segments[next_i].clone();
            let cl = client.clone();
            let cfg = cfg_snapshot.clone();
            let v = tts_voice.clone();
            handles.push_back(tokio::spawn(async move { tts(&cl, &cfg, &seg, &v).await }));
        }

        let handle = match handles.pop_front() {
            Some(h) => h,
            None => break,
        };

        let audio = match handle.await {
            Ok(Ok(a)) => a,
            Ok(Err(e)) => {
                error!("TTS segment {} failed: {}", i + 1, e);
                play_prompt(Prompt::Network, stop_playback).await;
                break;
            }
            Err(e) => {
                error!("TTS join failed: {}", e);
                break;
            }
        };

        let stop = stop_playback.clone();
        if let Err(e) = tokio::task::spawn_blocking(move || play_audio_bytes(audio, &stop)).await {
            error!("Playback failed: {}", e);
            break;
        }
    }

    // Drain anything that arrived while we were playing
    while rx.try_recv().is_ok() {}

    is_playing.store(false, Ordering::SeqCst);
    led.send(LedCommand::SetState(LedState::Idle));
}

// ---------------------------------------------------------------------------
// Alarm watcher
// ---------------------------------------------------------------------------

struct AlarmContext {
    store_path: String,
    groq: SharedGroq,
    client: reqwest::Client,
    is_playing: Arc<AtomicBool>,
    stop_playback: Arc<AtomicBool>,
    led: LedHandle,
}

async fn alarm_watcher(ctx: AlarmContext) {
    let mut last_fired: Vec<(String, String)> = Vec::new();
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;
        let now_hm = match local_time_hm() { Some(t) => t, None => continue };
        let today = local_date().unwrap_or_default();
        let weekday = local_weekday();
        let fire_key_suffix = format!("{} {}", today, now_hm);

        let mut alarms = load_alarms(&ctx.store_path);
        let mut dirty = false;
        let mut to_fire: Vec<(String, String, String)> = Vec::new();

        for alarm in alarms.iter_mut() {
            if !alarm.enabled { continue; }
            if alarm.time != now_hm { continue; }
            let key = (alarm.id.clone(), fire_key_suffix.clone());
            if last_fired.contains(&key) { continue; }
            let should_fire = match alarm.repeat.as_str() {
                "weekdays" => (1..=5).contains(&weekday),
                _ => true,
            };
            if !should_fire { continue; }
            last_fired.push(key);
            to_fire.push((alarm.id.clone(), alarm.time.clone(), alarm.label.clone()));
            if alarm.repeat == "once" { alarm.enabled = false; dirty = true; }
        }

        if dirty { save_alarms(&ctx.store_path, &alarms); }

        for (id, time, label) in to_fire {
            // Wait for any ongoing main-loop playback to finish
            while ctx.is_playing.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }

            let msg = if label.trim().is_empty() { format!("Alarm at {}", time) } else { format!("Alarm at {}: {}", time, label) };
            info!("Alarm firing ({}): {}", id, msg);
            ctx.led.send(LedCommand::SetState(LedState::Alarm));
            let cfg = ctx.groq.read().await.clone();
            match tts(&ctx.client, &cfg, &msg, &cfg.tts_voice).await {
                Ok(audio) => {
                    ctx.is_playing.store(true, Ordering::SeqCst);
                    let stop = ctx.stop_playback.clone();
                    let _ = tokio::task::spawn_blocking(move || { let _ = play_audio_bytes(audio, &stop); }).await;
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    ctx.is_playing.store(false, Ordering::SeqCst);
                    ctx.led.send(LedCommand::SetState(LedState::Idle));
                }
                Err(e) => warn!("Alarm TTS failed: {}", e),
            }
        }

        if last_fired.len() > 200 { let drop = last_fired.len() - 200; last_fired.drain(0..drop); }
    }
}

// ---------------------------------------------------------------------------
// Test commands
// ---------------------------------------------------------------------------

async fn test_mic(config: &AudioConfig, mic_gain: Arc<AtomicU32>) -> Result<()> {
    println!();
    println!("[Mic Test] Recording for 3 seconds, speak now...");
    let (tx, mut rx) = channel::<Vec<f32>>(100);
    let stream = start_capture(tx, config.sample_rate, mic_gain)?;
    let start = std::time::Instant::now();
    let mut max_rms: f32 = 0.0;
    let mut total_samples: usize = 0;
    use std::io::Write;
    while start.elapsed() < Duration::from_secs(3) {
        if let Ok(Some(chunk)) = tokio::time::timeout(Duration::from_millis(100), rx.recv()).await {
            let r = rms(&chunk);
            if r > max_rms { max_rms = r; }
            total_samples += chunk.len();
            let bar_len = ((r * 40.0).min(40.0)) as usize;
            let bar: String = "=".repeat(bar_len);
            let pad: String = " ".repeat(40 - bar_len);
            print!("\r  Level: [{}{}] {:.4}", bar, pad, r);
            std::io::stdout().flush().ok();
        }
    }
    drop(stream);
    println!();
    println!("  Samples captured: {}", total_samples);
    println!("  Peak RMS: {:.4}", max_rms);
    if max_rms > 0.01 { println!("  [OK] Microphone working"); }
    else if max_rms > 0.001 { println!("  [WARN] Very quiet input."); }
    else { println!("  [FAIL] No audio detected."); }
    Ok(())
}

fn test_speaker() -> Result<()> {
    println!();
    println!("[Speaker Test] Playing 440Hz tone for 2 seconds...");
    let sample_rate = 24000u32;
    let duration = 2.0f32;
    let freq = 440.0f32;
    let n = (sample_rate as f32 * duration) as usize;
    let samples: Vec<f32> = (0..n).map(|i| {
        let t = i as f32 / sample_rate as f32;
        (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3
    }).collect();
    let spec = WavSpec { channels: 1, sample_rate, bits_per_sample: 16, sample_format: HoundSampleFormat::Int };
    let mut cursor = Cursor::new(Vec::new());
    { let mut w = WavWriter::new(&mut cursor, spec)?; for &s in &samples { w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?; } w.finalize()?; }
    let bytes = cursor.into_inner();
    let stop = Arc::new(AtomicBool::new(false));
    play_audio_bytes(bytes, &stop)?;
    println!("  [OK] Speaker test done.");
    Ok(())
}

async fn test_volume(config: &AudioConfig) -> Result<()> {
    println!();
    println!("[Volume Test]");
    match amixer_get(config).await { Ok(v) => println!("  Current: {}%", v), Err(e) => println!("  Cannot read: {}", e) }
    for pct in [20u32, 80, 50] {
        match amixer_set(config, pct).await { Ok(a) => println!("  Set to {}%", a), Err(e) => println!("  Failed: {}", e) }
        tokio::time::sleep(Duration::from_millis(800)).await;
    }
    println!("  [OK] Volume test done.");
    Ok(())
}

fn test_keys(config: &KeysConfig) -> Result<()> {
    if !config.enabled { println!(); println!("[Keys Test] Disabled in config"); return Ok(()); }
    if !PathBuf::from(&config.device).exists() { println!(); println!("[Keys Test] Device {} not found", config.device); return Ok(()); }
    println!();
    println!("[Keys Test] Listening on {} for 15 seconds", config.device);
    use std::io::Read;
    use std::os::unix::io::AsRawFd;
    let mut file = std::fs::File::open(&config.device)?;
    unsafe {
        let flags = libc::fcntl(file.as_raw_fd(), libc::F_GETFL, 0);
        libc::fcntl(file.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK);
    }
    let start = std::time::Instant::now();
    let mut buf = [0u8; 16];
    while start.elapsed() < Duration::from_secs(15) {
        match file.read(&mut buf) {
            Ok(16) => {
                let ev_type = u16::from_le_bytes([buf[8], buf[9]]);
                let ev_code = u16::from_le_bytes([buf[10], buf[11]]);
                let ev_value = i32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
                if ev_type == 1 && ev_value == 1 {
                    let name = if ev_code == config.mute { " (mute)" }
                    else if ev_code == config.volume_up { " (vol+)" }
                    else if ev_code == config.volume_down { " (vol-)" }
                    else if ev_code == config.play_pause { " (play/pause)" }
                    else { "" };
                    println!("  Key code: {}{}", ev_code, name);
                }
            }
            _ => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    println!("  [OK] Keys test done");
    Ok(())
}

async fn test_network(client: &reqwest::Client, proxy_url: &str) -> Result<()> {
    println!();
    println!("[Network Test]");
    println!("  Proxy: {}", if proxy_url.is_empty() { "none" } else { proxy_url });
    match client.get(IP_API_URL).send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            if status == 200 {
                if let Ok(info) = resp.json::<IpApiResponse>().await {
                    println!("  [OK] IP: {} ({}, {})", info.query.as_deref().unwrap_or("?"), info.city.as_deref().unwrap_or("?"), info.country.as_deref().unwrap_or("?"));
                }
            } else { println!("  [WARN] HTTP {}", status); }
        }
        Err(e) => println!("  [FAIL] {}", e),
    }
    Ok(())
}

async fn test_groq(client: &reqwest::Client, config: &GroqConfig) -> Result<()> {
    println!();
    println!("[Groq Test]");
    if config.api_key.trim().is_empty() { println!("  [FAIL] API key not set"); return Ok(()); }
    match client.get(MODELS_URL).bearer_auth(&config.api_key).send().await {
        Ok(r) if r.status().is_success() => {
            println!("  [OK] API key valid");
            if let Ok(models) = r.json::<ModelsResponse>().await {
                let ids: Vec<String> = models.data.iter().map(|m| m.id.clone()).collect();
                for (label, id) in [("STT", &config.stt_model), ("LLM", &config.llm_model), ("TTS", &config.tts_model)] {
                    if ids.iter().any(|i| i == id) { println!("  [OK] {} model: {}", label, id); }
                    else { println!("  [WARN] {} model not available: {}", label, id); }
                }
            }
        }
        Ok(r) => println!("  [FAIL] API key: HTTP {}", r.status()),
        Err(e) => println!("  [FAIL] Cannot reach: {}", e),
    }
    println!("  Testing TTS synthesis...");
    match tts(client, config, "Test.", "autumn").await {
        Ok(audio) => println!("  [OK] TTS returned {} bytes", audio.len()),
        Err(e) => println!("  [FAIL] TTS: {}", e),
    }
    Ok(())
}

fn test_led(config: &LedConfig) -> Result<()> {
    println!();
    println!("[LED Test]");
    if !config.enabled { println!("  [SKIP] LED disabled in config"); return Ok(()); }
    println!("  Device: {}", config.device);
    if !PathBuf::from(&config.device).exists() { println!("  [FAIL] Device not found"); return Ok(()); }

    println!();
    println!("  Phase 1: full ring primary colors");
    let primaries: [(u32, &str); 4] = [
        (0x0000FF, "red"), (0x00FF00, "green"), (0xFF0000, "blue"), (0xFFFFFF, "white"),
    ];
    for (bgr, name) in primaries {
        println!("    -> all LEDs = {}", name);
        let arr = [bgr; LED_COUNT];
        if let Err(e) = write_leds(&config.device, &arr) { println!("    [FAIL] {}", e); return Ok(()); }
        std::thread::sleep(Duration::from_millis(600));
    }

    println!();
    println!("  Phase 2: individual LED scan (0 to {})", LED_COUNT - 1);
    for i in 0..LED_COUNT {
        let mut arr = [0u32; LED_COUNT];
        arr[i] = 0xFFFFFF;
        if let Err(e) = write_leds(&config.device, &arr) { println!("    [FAIL] at LED {}: {}", i, e); return Ok(()); }
        print!("\r    LED {}/{}", i, LED_COUNT - 1);
        use std::io::Write;
        std::io::stdout().flush().ok();
        std::thread::sleep(Duration::from_millis(120));
    }
    println!();

    println!();
    println!("  Phase 3: brightness ramp");
    let base = parse_rgb_hex_to_bgr(&config.idle_color).unwrap_or(0x0000C8);
    for pct in [10u32, 30, 50, 70, 90, 100, 70, 40, 10] {
        let scaled = apply_brightness(base, pct as u8);
        let arr = [scaled; LED_COUNT];
        let _ = write_leds(&config.device, &arr);
        print!("\r    brightness {}%", pct);
        use std::io::Write;
        std::io::stdout().flush().ok();
        std::thread::sleep(Duration::from_millis(200));
    }
    println!();

    println!();
    println!("  Phase 4: preset states");
    let states: [(u32, &str); 4] = [
        (state_to_color(LedState::Idle, config), "idle"),
        (state_to_color(LedState::Listening, config), "listening"),
        (state_to_color(LedState::Speaking, config), "speaking"),
        (state_to_color(LedState::Muted, config), "muted"),
    ];
    for (bgr, name) in states {
        println!("    -> state {}", name);
        let arr = [bgr; LED_COUNT];
        let _ = write_leds(&config.device, &arr);
        std::thread::sleep(Duration::from_millis(700));
    }

    println!();
    println!("  Phase 5: fade to black");
    let start = parse_rgb_hex_to_bgr(&config.idle_color).unwrap_or(0x0000C8);
    for step in (0..=10).rev() {
        let pct = (step * 10) as u8;
        let scaled = apply_brightness(start, pct);
        let arr = [scaled; LED_COUNT];
        let _ = write_leds(&config.device, &arr);
        std::thread::sleep(Duration::from_millis(80));
    }
    let off = [0u32; LED_COUNT];
    let _ = write_leds(&config.device, &off);

    println!();
    println!("  [OK] LED test complete");
    println!("  Expected: 18 LEDs scanned one by one, brightness ramp, 4 preset states, fade out");
    Ok(())
}

fn test_alarms(store_path: &str) -> Result<()> {
    println!();
    println!("[Alarms Test]");
    println!("  Store path: {}", store_path);
    for line in list_alarms_text(store_path).split("; ") { println!("    {}", line); }
    Ok(())
}

async fn run_test_command(subcommand: &str, config: &Config, client: &reqwest::Client, mic_gain: Arc<AtomicU32>) -> Result<()> {
    println!();
    println!("============================================================");
    println!("  Rimth Hardware Test: {}", subcommand);
    println!("============================================================");
    match subcommand {
        "mic" => test_mic(&config.audio, mic_gain).await?,
        "speaker" => test_speaker()?,
        "volume" => test_volume(&config.audio).await?,
        "keys" => test_keys(&config.keys)?,
        "network" => test_network(client, &config.proxy.url).await?,
        "groq" => test_groq(client, &config.groq).await?,
        "led" => test_led(&config.led)?,
        "alarms" => test_alarms(&config.alarm.store_path)?,
        "all" | "" => {
            test_mic(&config.audio, mic_gain).await?;
            test_speaker()?;
            test_volume(&config.audio).await?;
            test_network(client, &config.proxy.url).await?;
            test_groq(client, &config.groq).await?;
            test_led(&config.led)?;
            test_keys(&config.keys)?;
            test_alarms(&config.alarm.store_path)?;
        }
        other => { eprintln!("Unknown test: {}", other); eprintln!("Available: mic, speaker, volume, keys, network, groq, led, alarms, all"); std::process::exit(1); }
    }
    println!();
    println!("============================================================");
    println!("  Test complete");
    println!("============================================================");
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

fn load_config_or_default() -> Config {
    let path = match config_path() {
        Ok(p) => p,
        Err(e) => { warn!("Cannot resolve config path: {}, using defaults", e); return Config::default(); }
    };
    if !path.exists() { warn!("Config not found at {}, using defaults", path.display()); return Config::default(); }
    match std::fs::read_to_string(&path) {
        Ok(content) => match toml::from_str::<Config>(&content) {
            Ok(cfg) => { info!("Loaded config from: {}", path.display()); cfg }
            Err(e) => { warn!("Parse error: {}, using defaults", e); Config::default() }
        },
        Err(e) => { warn!("Read error: {}, using defaults", e); Config::default() }
    }
}

fn write_default_config() -> Result<()> {
    let path = config_path()?;
    let toml_str = toml::to_string_pretty(&Config::default())?;
    std::fs::write(&path, toml_str)?;
    println!("Config written to: {}", path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

struct KeyState {
    muted: Arc<AtomicBool>,
    stop_playback: Arc<AtomicBool>,
    conversation_mode: Arc<AtomicBool>,
}

struct AppState {
    key_state: Arc<KeyState>,
    sessions_dir: PathBuf,
    mic_gain: Arc<AtomicU32>,
    led: LedHandle,
    groq: SharedGroq,
    messages: StdMutex<Vec<ChatMessage>>,
    current_session: StdMutex<String>,
    effective_system: StdMutex<String>,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--write-config") { return write_default_config(); }

    let config = load_config_or_default();
    let client = build_client(&config.proxy.url)?;
    if !config.proxy.url.is_empty() { info!("Using proxy: {}", config.proxy.url); }

    let mic_gain = Arc::new(AtomicU32::new((config.audio.mic_gain).to_bits()));

    if args.len() >= 2 && args[1] == "test" {
        let sub = args.get(2).map(|s| s.as_str()).unwrap_or("all");
        return run_test_command(sub, &config, &client, mic_gain).await;
    }

    let groq: SharedGroq = Arc::new(RwLock::new(config.groq.clone()));
    let led = spawn_led_worker(config.led.clone());

    let key_state = Arc::new(KeyState {
        muted: Arc::new(AtomicBool::new(false)),
        stop_playback: Arc::new(AtomicBool::new(false)),
        conversation_mode: Arc::new(AtomicBool::new(false)),
    });

    start_key_listener(config.keys.clone(), config.audio.clone(), led.clone(), key_state.clone());

    if config.diagnostics.run_on_startup || args.iter().any(|a| a == "--diagnostics") {
        let ok = run_diagnostics(&config, &client).await;
        if !ok { warn!("Diagnostics reported failures"); }
    }

    {
        let cfg = groq.read().await;
        if !cfg.api_key.trim().is_empty() {
            if let Err(e) = generate_prompt_sounds(&client, &cfg).await {
                warn!("Prompt sound generation failed: {}", e);
            }
        }
    }

    let sessions_dir = if config.session.persist_dir.is_empty() { sessions_dir()? }
    else { PathBuf::from(&config.session.persist_dir) };
    std::fs::create_dir_all(&sessions_dir)?;

    let effective_system = {
        let cfg = groq.read().await;
        build_system_prompt(&cfg.system_prompt, &cfg.language, &cfg)
    };

    let current_session = "default".to_string();
    let mut initial_messages: Vec<ChatMessage> = load_session(&sessions_dir, &current_session);
    if initial_messages.is_empty() || initial_messages[0].role != "system" {
        initial_messages.insert(0, ChatMessage::system(effective_system.clone()));
    } else {
        initial_messages[0].content = Some(effective_system.clone());
    }

    let is_playing = Arc::new(AtomicBool::new(false));
    let (tx, mut rx) = channel::<Vec<f32>>(100);
    let capture = CaptureCtx::new(tx.clone(), config.audio.sample_rate, mic_gain.clone())?;

    let app_state = Arc::new(AppState {
        key_state: key_state.clone(),
        sessions_dir: sessions_dir.clone(),
        mic_gain: mic_gain.clone(),
        led: led.clone(),
        groq: groq.clone(),
        messages: StdMutex::new(initial_messages),
        current_session: StdMutex::new(current_session.clone()),
        effective_system: StdMutex::new(effective_system.clone()),
    });

    {
        let cfg = groq.read().await;
        if !cfg.api_key.trim().is_empty() {
            let alarm_ctx = AlarmContext {
                store_path: config.alarm.store_path.clone(),
                groq: groq.clone(),
                client: client.clone(),
                is_playing: is_playing.clone(),
                stop_playback: key_state.stop_playback.clone(),
                led: led.clone(),
            };
            tokio::spawn(alarm_watcher(alarm_ctx));
            info!("Alarm watcher started ({} alarms loaded)", load_alarms(&config.alarm.store_path).len());
        }
    }

    led.send(LedCommand::SetState(LedState::Idle));

    let lang_now = groq.read().await.language.clone();
    info!("Rimth listening... (session: {}, lang: {})", current_session, lang_now);

    loop {
        if key_state.muted.load(Ordering::SeqCst) {
            while rx.try_recv().is_ok() {}
            tokio::time::sleep(Duration::from_millis(200)).await;
            continue;
        }

        let trigger_chunk = if key_state.conversation_mode.load(Ordering::Relaxed) {
            info!("Conversation mode active, listening...");
            led.send(LedCommand::SetState(LedState::Listening));
            tokio::time::sleep(Duration::from_millis(300)).await;
            while rx.try_recv().is_ok() {}
            Vec::new()
        } else {
            match wait_trigger(&mut rx, &config.audio, &is_playing, &key_state, &capture).await {
                Some(c) => c,
                None => continue,
            }
        };
        led.send(LedCommand::SetState(LedState::Listening));

        let api_key_now = groq.read().await.api_key.clone();
        if api_key_now.trim().is_empty() { warn!("API key not set, skipping"); continue; }

        info!("Recording...");
        let wav_opt = match record(&mut rx, &config.audio, &is_playing, &key_state, trigger_chunk).await {
            Ok(w) => w,
            Err(e) => { error!("Record failed: {}", e); continue; }
        };
        let wav = match wav_opt {
            Some(w) => w,
            None => { led.send(LedCommand::SetState(LedState::Idle)); continue; }
        };

        if wav.len() < MIN_AUDIO_BYTES {
            info!("Recording too short ({} bytes), skipping", wav.len());
            led.send(LedCommand::SetState(LedState::Idle));
            continue;
        }

        info!("Transcribing...");
        let user_text = {
            let cfg = groq.read().await.clone();
            match stt(&client, &cfg, wav).await {
                Ok(t) => t,
                Err(e) => {
                    warn!("STT {}: {}", e.status, e.message);
                    let p = match e.status {
                        401 | 403 => Prompt::Auth, 429 | 413 => Prompt::RateLimit,
                        400 | 422 => Prompt::BadRequest, 500 | 502 | 503 => Prompt::Server,
                        _ => Prompt::Server,
                    };
                    play_prompt(p, &key_state.stop_playback).await;
                    led.send(LedCommand::SetState(LedState::Idle));
                    continue;
                }
            }
        };

        info!("User: {}", user_text);
        if user_text.trim().is_empty() { led.send(LedCommand::SetState(LedState::Idle)); continue; }
        if is_hallucination(&user_text) {
            warn!("Detected Whisper hallucination, skipping: {}", user_text);
            led.send(LedCommand::SetState(LedState::Idle));
            continue;
        }

        let lower = user_text.trim().to_lowercase();

        if lower == "diagnostics" || lower == "run diagnostics" {
            let ok = run_diagnostics(&config, &client).await;
            let reply = if ok { "Diagnostics passed." } else { "Diagnostics found problems." };
            let cfg = groq.read().await.clone();
            speak(&client, &cfg, reply, &is_playing, &key_state.stop_playback, &mut rx, &led).await;
            continue;
        }

        // Regenerate the system prompt every turn so model / language changes
        // are reflected immediately.
        let effective_system = {
            let cfg = groq.read().await;
            build_system_prompt(&cfg.system_prompt, &cfg.language, &cfg)
        };
        {
            let mut sys = app_state.effective_system.lock().unwrap();
            *sys = effective_system.clone();
        }
        {
            let mut msgs = app_state.messages.lock().unwrap();
            if msgs.is_empty() || msgs[0].role != "system" {
                msgs.insert(0, ChatMessage::system(effective_system.clone()));
            } else {
                msgs[0].content = Some(effective_system.clone());
            }
        }

        if lower.starts_with("new session") {
            let name = lower.trim_start_matches("new session").trim().to_string();
            let name = if name.is_empty() { format!("session_{}", rand::random::<u16>()) } else { name };
            {
                let mut cur = app_state.current_session.lock().unwrap();
                let msgs = app_state.messages.lock().unwrap().clone();
                save_session(&sessions_dir, &cur, &msgs);
                *cur = name.clone();
            }
            {
                let mut msgs = app_state.messages.lock().unwrap();
                msgs.clear();
                msgs.push(ChatMessage::system(effective_system.clone()));
            }
            let reply = format!("New session: {}", name);
            info!("{}", reply);
            let cfg = groq.read().await.clone();
            speak(&client, &cfg, &reply, &is_playing, &key_state.stop_playback, &mut rx, &led).await;
            continue;
        }

        if lower.starts_with("switch session") || lower.starts_with("resume session") {
            let prefix = if lower.starts_with("switch session") { "switch session" } else { "resume session" };
            let name = lower.trim_start_matches(prefix).trim().to_string();
            let sessions = list_sessions(&sessions_dir);
            if name.is_empty() || !sessions.contains(&name) {
                let reply = if sessions.is_empty() { "No saved sessions".to_string() }
                else { format!("Sessions: {}", sessions.join(", ")) };
                info!("{}", reply);
                let cfg = groq.read().await.clone();
                speak(&client, &cfg, &reply, &is_playing, &key_state.stop_playback, &mut rx, &led).await;
                continue;
            }
            {
                let mut cur = app_state.current_session.lock().unwrap();
                let msgs = app_state.messages.lock().unwrap().clone();
                save_session(&sessions_dir, &cur, &msgs);
                *cur = name.clone();
            }
            let mut loaded = load_session(&sessions_dir, &name);
            if loaded.is_empty() || loaded[0].role != "system" {
                loaded.insert(0, ChatMessage::system(effective_system.clone()));
            }
            {
                let mut msgs = app_state.messages.lock().unwrap();
                *msgs = loaded;
            }
            let reply = format!("Resumed: {}", name);
            info!("{}", reply);
            let cfg = groq.read().await.clone();
            speak(&client, &cfg, &reply, &is_playing, &key_state.stop_playback, &mut rx, &led).await;
            continue;
        }

        if lower == "list sessions" {
            let sessions = list_sessions(&sessions_dir);
            let reply = if sessions.is_empty() { "No sessions".to_string() }
            else { format!("Sessions: {}", sessions.join(", ")) };
            info!("{}", reply);
            let cfg = groq.read().await.clone();
            speak(&client, &cfg, &reply, &is_playing, &key_state.stop_playback, &mut rx, &led).await;
            continue;
        }

        // Push user message and trim
        {
            let mut msgs = app_state.messages.lock().unwrap();
            msgs.push(ChatMessage::user(user_text.clone()));
            trim_history(&mut msgs, config.session.max_history);
        }

        info!("Thinking...");
        led.send(LedCommand::StartThinking);
        let tools = tools_definition();
        let mut assistant_text = String::new();

        for _ in 0..5 {
            let cfg = groq.read().await.clone();
            let msgs_snapshot: Vec<ChatMessage> = {
                let msgs = app_state.messages.lock().unwrap();
                msgs.clone()
            };

            match chat(&client, &cfg, &msgs_snapshot, Some(tools.clone())).await {
                Ok(response) => {
                    if let Some(tool_calls) = &response.tool_calls {
                        if !tool_calls.is_empty() {
                            {
                                let mut msgs = app_state.messages.lock().unwrap();
                                msgs.push(ChatMessage::assistant(response.content.clone(), Some(tool_calls.clone())));
                            }
                            for tc in tool_calls {
                                let result = execute_tool(
                                    &client, &config.audio, &config.alarm, &config,
                                    &app_state, &tc.function.name, &tc.function.arguments
                                ).await;
                                info!("Tool ({}): {}", tc.function.name, result);
                                {
                                    let mut msgs = app_state.messages.lock().unwrap();
                                    msgs.push(ChatMessage::tool(tc.id.clone(), tc.function.name.clone(), result));
                                }
                            }
                            continue;
                        }
                    }
                    assistant_text = response.content.unwrap_or_default();
                    break;
                }
                Err(e) => {
                    error!("Chat failed: {}", e);
                    let msg = e.to_string();
                    let p = if msg.contains("429") || msg.contains("413") { Prompt::RateLimit }
                    else if msg.contains("401") || msg.contains("403") { Prompt::Auth }
                    else if msg.contains("400") || msg.contains("422") { Prompt::BadRequest }
                    else { Prompt::Server };
                    led.send(LedCommand::StopThinking);
                    play_prompt(p, &key_state.stop_playback).await;
                    if msg.contains("Tools should have a name") || msg.contains("tool_call_id") {
                        warn!("History corrupted, sanitizing");
                        let mut msgs = app_state.messages.lock().unwrap();
                        let cleaned = sanitize_session(msgs.clone());
                        *msgs = cleaned;
                    }
                    break;
                }
            }
        }

        led.send(LedCommand::StopThinking);

        if assistant_text.trim().is_empty() { continue; }
        info!("Assistant: {}", assistant_text);
        {
            let mut msgs = app_state.messages.lock().unwrap();
            msgs.push(ChatMessage::assistant(Some(assistant_text.clone()), None));
            trim_history(&mut msgs, config.session.max_history);
        }

        // Persist
        {
            let cur = app_state.current_session.lock().unwrap().clone();
            let msgs = app_state.messages.lock().unwrap().clone();
            save_session(&sessions_dir, &cur, &msgs);
        }

        info!("Synthesizing...");
        let cfg = groq.read().await.clone();
        speak(&client, &cfg, &assistant_text, &is_playing, &key_state.stop_playback, &mut rx, &led).await;
    }
}