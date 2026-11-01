# Meme Bot

A live speech-to-text bot that listens to your microphone, types what you say in real time, and plays a fitting meme sound effect through your speakers.

- **Whisper** (faster-whisper) turns your voice into text, word by word.
- **d1** (a decision model that runs in Ollama) reads each finished sentence and picks the best meme sound for it.
- **Talking over a sound stops it** (after a short protected window).

---

## How it works

1. You speak. Words appear live in the console once Whisper is sure of them.
2. When you pause, the sentence is finalized with a more accurate pass.
3. d1 answers two questions about the sentence:
   - Is this a good moment for a meme sound? (the **meme-worthy** score)
   - Which sound in your `sfx` folder fits best? (the **best** sound, with a confidence)
4. If the score passes the gate and the cooldown has passed, the sound plays on your speakers.
5. If you talk again (after the first 2 seconds), the sound stops and the bot listens to you.

---

## Requirements

- Windows PC with a microphone and speakers
- Python (with a virtual environment, `.venv`)
- An NVIDIA GPU is recommended. Without one, the bot falls back to the CPU automatically (slower).
- [Ollama](https://ollama.com/download), with the `d1` model pulled

---

## Setup (one time)

### 1. Install Ollama and the d1 model
1. Download and install Ollama from https://ollama.com/download (update it if you already have it).
2. Make sure it is running (llama icon in the system tray).
3. In Command Prompt:
   ```
   ollama pull d1
   ```
   (about 3.7 GB)

### 2. Install the Python packages
From the project folder, using the venv's Python:
```
.venv\Scripts\python.exe -m pip install --upgrade faster-whisper sounddevice numpy requests av nvidia-cublas-cu12 nvidia-cudnn-cu12 pygame-ce soundfile
```
- `pygame-ce` plays the sounds. If it fails to install, `soundfile` + `sounddevice` is used automatically instead.
- Do **not** install plain `pygame` on a very new Python version. It has no prebuilt package and fails to compile. Use `pygame-ce`.

### 3. Add your sounds
Create a folder called `sfx` next to `main.py`:
```
meme bot\
    main.py
    README.md
    sfx\
        oof.mp3
        vine_boom.mp3
        sad_violin.mp3
        ...
```
- Formats: `.mp3`, `.wav`, `.ogg`
- The file name (without the extension) is the sound's name, so use simple names with underscores.

### 4. Describe each sound
In `main.py`, find the `DESCRIPTIONS` block and add one line per sound:
```python
"oof": "failure, bad news, failing an exam, something going wrong",
```
The key must match the file name exactly (without the extension). d1 chooses sounds based only on these descriptions, so describe the situations where the sound fits. A file with no description falls back to its file name.

---

## Run it

1. Make sure Ollama is running.
2. In the project folder:
   ```
   .venv\Scripts\python.exe main.py
   ```
   (or press Run in PyCharm with the `.venv` interpreter selected)

Startup order: Whisper loads, d1 warms up (the first run can take about a minute), then you see `Ready!`. Talk, and press Ctrl+C to stop.

The first run also downloads the Whisper model (`large-v3-turbo`, about 1.5 GB).

---

## Settings (top of `main.py`)

| Setting | What it does |
|---|---|
| `LANGUAGE` | Language you speak: `"en"`, `"fr"`, `"ar"`, or `None` for auto-detect (slower, better for mixed languages) |
| `MODEL_SIZE` | Whisper model: `"large-v3-turbo"` (best), `"medium"`, `"small"` (fastest). Use a smaller one if d1 is slow or the GPU is full |
| `SPEECH_THRESHOLD` | Mic level that counts as speech. Raise it if noise triggers recordings, lower it if it misses you |
| `SILENCE_END_SEC` | Pause length that ends a sentence |
| `PARTIAL_EVERY_SEC` | How often live words are refreshed |
| `GATE_THRESHOLD` | Minimum meme-worthy score needed to play a sound. `0.0` = a sound for every sentence, `0.7` = only obvious moments |
| `MIN_CONFIDENCE` | Minimum d1 confidence needed to play (raise to `0.3` for fewer wrong picks) |
| `COOLDOWN_SEC` | Minimum time between two sounds |
| `VOLUME` | Sound volume, `0.0` to `1.0` |
| `OUTPUT_DEVICE` | `None` = default speakers. Or a device number from `sd.query_devices()` |
| `BARGE_IN_THRESHOLD` | Voice level that stops a playing sound |
| `BARGE_IN_BLOCKS` | How long you must talk to stop a sound (4 = about 0.2 s) |
| `MIN_PLAY_SEC` | A sound always plays at least this long before it can be interrupted |
| `PROMPT` | A hint to Whisper about your vocabulary (slang, names, topics) |

---

## What the console shows

```
I failed my math exam
  meme-worthy=0.82  best=oof  confidence=0.74
  >> playing oof
```
If a sound is not played, it tells you why:
```
  -- skipped: meme-worthy 0.21 < 0.3
  -- skipped: confidence 0.10 < 0.3
  -- skipped: cooldown (1.2s < 2.0s)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No sounds found` | The `sfx` folder is missing, misnamed, or empty |
| `d1 error: Connection refused` | Ollama is not running. Start it |
| `d1 error: 404` / model not found | Run `ollama pull d1`, or update Ollama |
| `d1 error: Read timed out` | d1 is still loading. Wait for the warm-up. If it keeps happening, run `ollama ps`: if d1 is on CPU, use a smaller Whisper model to free GPU memory |
| `Library cublas64_12.dll is not found` | Install `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` (included above). The `os.add_dll_directory` block at the top of `main.py` must stay |
| `GPU failed ... Falling back to CPU` | It still works but is slower. The message after "GPU failed" tells you why |
| `TypeError: open() got an unexpected keyword argument 'metadata_errors'` | Old `av` package. Run `pip install --upgrade av` with the venv's Python |
| `pygame` fails to build | Install `pygame-ce` instead |
| Wrong words or phrases like "see you next time" from silence | Whisper hallucinations. Raise `SPEECH_THRESHOLD`, use a better mic or headset, or add the phrase to `HALLUCINATIONS` |
| First syllable of sentences is missing | Raise `PREROLL_SEC` (default 0.4) |
| Sound is picked but never plays | Read the `-- skipped:` line. Usually `GATE_THRESHOLD` is too high |
| Sound plays but you hear nothing | Check the Windows output device and volume, or set `OUTPUT_DEVICE` |
| A sound cuts itself off | The mic hears your speakers. Raise `BARGE_IN_THRESHOLD`, lower the volume, or use headphones |
| A sound does not stop when you talk | Lower `BARGE_IN_THRESHOLD` (e.g. `0.012`) or `BARGE_IN_BLOCKS`, and remember `MIN_PLAY_SEC` |
| It doesn't hear you | Check the Windows input device and that desktop apps may use the microphone (Settings, Privacy, Microphone) |

### Find your audio devices
```
.venv\Scripts\python.exe -c "import sounddevice as sd; print(sd.query_devices())"
```

---

## Tips for good sound picks

- Start with 10 to 15 sounds. d1 picks more reliably with a smaller set.
- Write descriptions as situations ("failing an exam, bad news"), not just feelings.
- Tune `GATE_THRESHOLD` using the `meme-worthy=` numbers shown in the console: set it between the scores of sentences that should and should not get a sound.
- Use headphones if you want to talk over sounds without the mic hearing the speakers.

---

## Notes

- All speech recognition and d1 run locally on your PC. Nothing is sent to the internet except the one-time model downloads.
- Check the license or terms of any sound files you download, especially if you share or stream the result.
