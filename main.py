import os, site, glob, time, re

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

for sp in site.getsitepackages():
    for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

import queue
from collections import deque
import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

# ---------------- settings ----------------
SAMPLE_RATE = 16000
BLOCK_SEC = 0.05
SPEECH_THRESHOLD = 0.006     # raise if noise triggers it, lower if it misses you
SILENCE_END_SEC = 0.8        # pause that ends a sentence
PARTIAL_EVERY_SEC = 0.4      # how often live words are checked
TRIM_AFTER_SEC = 8
MAX_UTTERANCE_SEC = 60
PREROLL_SEC = 0.4
LANGUAGE = "en"              # "en", "fr", "ar", or None for auto
MODEL_SIZE = "large-v3-turbo"   # use "medium" or "small" if d1 is slow / GPU is full
PROMPT = "Casual conversation, slang, memes, school, exams."

SFX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sfx")
OLLAMA_URL = "http://localhost:11434/v1/systemone"
GATE_THRESHOLD = 0.0         # lower = more sounds
MIN_CONFIDENCE = 0.0
COOLDOWN_SEC = 2.0
VOLUME = 1.0                 # 0.0 to 1.0
OUTPUT_DEVICE = None         # None = default speakers. Or a device number/name from sd.query_devices()

# Talking over a playing sound stops it:
BARGE_IN_THRESHOLD = 0.02    # voice level that stops a sound (higher than SPEECH_THRESHOLD
                             # so the sound doesn't cut itself off via the mic)
BARGE_IN_BLOCKS = 4          # how many 50 ms blocks in a row (~0.2 s)
MIN_PLAY_SEC = 2.0           # a sound always plays at least this long before you can interrupt it

DESCRIPTIONS = {
    # failure / bad news
    "noo_la_policia" : "six seven, brainrot, triple t, tung tung tung sahur",
    "oof": "failure, bad news, getting hurt, bad grade, failing an exam, something going wrong",
    "sad_trombone": "a small fail, disappointment, plans falling apart, wah wah",
    "windows_error": "a mistake, something broke, a technical error, a bug or crash",
    "gta_wasted": "dying, getting destroyed, being completely done for",
    "game_over": "total failure, the end, giving up, losing everything",
    "buzzer_wrong": "a wrong answer, an incorrect statement, being corrected",
    "price_is_right_fail": "losing, missing out, a pathetic failed attempt",
    "emotional_damage": "an insult, hurt feelings, a harsh roast, being offended",
    "sad_violin": "sadness, self-pity, a tragic or pathetic situation, complaining about life",
    "spongebob_disappointed": "deep disappointment, being let down, sighing at something bad",
    # shock / drama
    "vine_boom": "a dramatic reveal, shocking statement, sudden plot twist, an exaggerated claim",
    "taco_bell_bong": "a sudden dramatic moment, ominous surprise, something suspicious happening",
    "dun_dun_dun": "suspense, a bad omen, something dangerous is coming",
    "dramatic_chipmunk": "a dramatic turn to the camera, shocking realization, a big reveal",
    "wilhelm_scream": "falling, being thrown, a dramatic fall or crash",
    "omg": "sfailure, bad news, getting hurt, bad grade, failing an exam, something going wrong",
    "record_scratch": "an interruption, wait what, freeze frame, an unexpected record-scratch moment",
    # disbelief / cringe
    "bruh": "disappointment, something dumb or unbelievable, a silly mistake",
    "nope": "refusing, no way, rejecting an idea, saying no",
    "curb_enthusiasm": "an awkward situation, an uncomfortable outcome, things going slightly wrong",
    "crickets": "awkward silence, a joke nobody laughed at, no response",
    "goofy_laugh": "something silly, a goofy dumb moment, laughing at a mistake",
    # win / hype
    "airhorn": "a big win, hype, celebration, excitement, a huge success",
    "victory_fanfare": "passing an exam, an achievement, finishing a task, a major accomplishment",
    "tada": "a successful reveal, here it is, finishing something nicely",
    "siuuu": "celebration, scoring a goal, football, a big hype moment",
    "sheesh": "praise, something impressive or cool, looking great, compliments",
    "wow": "amazement, admiring something impressive",
    "applause": "congratulations, well done, praising someone's effort",
    "ding": "a correct answer, getting something right, a small success",
    "okey_lets_go" : "let's go",
    # caught / suspicious
    "fbi_open_up": "something illegal, cheating, doing something wrong, getting caught by police",
    "among_us": "suspicion, accusing someone, someone sneaky or lying, being sus",
    "metal_gear_alert": "being noticed, getting caught, sudden danger, a moment of panic",
    "john_cena": "a sudden surprise appearance, invisibility, you can't see me, out of nowhere",
    # jokes / misc
    "rimshot": "a joke or punchline, a bad pun, dad joke",
    "laugh_track": "something funny, a sitcom moment, a joke that landed",
    "bonk": "punishing someone for a silly idea, stop being silly, going to meme jail",
    "yeet": "throwing something, getting rid of something, an energetic launch",
    "cash_register": "money, getting paid, buying something, a high price, being rich or broke",
    "windows_xp_shutdown": "leaving, going to sleep, saying goodbye, ending the day",
    "why_are_you_running": "running away, escaping, being in a hurry, avoiding something",
    "explosion": "an explosion, a huge disaster, everything blowing up, a massive drop",
}

HALLUCINATIONS = {"thank you", "thanks for watching", "see you next time",
                  "and ill see you next time", "bye", "you"}
# ------------------------------------------

# ---------- sounds (played through your speakers) ----------
sfx = {}
for path in glob.glob(os.path.join(SFX_DIR, "*")):
    name, ext = os.path.splitext(os.path.basename(path))
    if ext.lower() in (".mp3", ".wav", ".ogg"):
        sfx[name] = path
if not sfx:
    raise SystemExit(f"No sounds found. Put .mp3/.wav/.ogg files in: {SFX_DIR}")
print("Sounds loaded:", ", ".join(sfx))


class PygamePlayer:
    def __init__(self, path):
        self.snd = pygame.mixer.Sound(path)
        self.snd.set_volume(VOLUME)
        self.length = self.snd.get_length()

    def play(self):
        self.snd.play()

    def stop(self):
        self.snd.stop()


class SDPlayer:
    def __init__(self, path):
        data, self.sr = sf.read(path, dtype="float32")
        self.data = data * VOLUME
        self.length = len(data) / self.sr

    def play(self):
        sd.play(self.data, self.sr, device=OUTPUT_DEVICE)   # non-blocking

    def stop(self):
        sd.stop()


sound_cache = None
if OUTPUT_DEVICE is None:
    try:
        import pygame
        pygame.mixer.init()
        sound_cache = {n: PygamePlayer(p) for n, p in sfx.items()}
        print("Audio player: pygame (default speakers)")
    except Exception:
        sound_cache = None
if sound_cache is None:
    import soundfile as sf
    sound_cache = {n: SDPlayer(p) for n, p in sfx.items()}
    print("Audio player: sounddevice + soundfile")

criteria = {n: DESCRIPTIONS.get(n, n.replace("_", " ")) for n in sfx}


# ---------- d1 ----------
def pick_sfx(text):
    payload = {
        "model": "d1",
        "state": text,
        "keep_alive": "60m",
        "questions": {
            "worth_it": {
                "type": "noul",
                "instructions": "Is this a moment where a meme sound effect reaction would be funny or fitting?",
            },
            "best_sfx": {
                "type": "choice",
                "instructions": "Which meme sound effect is the best reaction to what the speaker just said?",
                "criteria": criteria,
            },
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    ans = r.json()["answers"]
    return (ans["worth_it"]["noul"], ans["best_sfx"]["choice"],
            ans["best_sfx"].get("confidence", 1.0))


last_play = 0.0
mute_until = 0.0         # while a sound plays we only listen for barge-in
current_player = None
bargein_count = 0


def handle(text):
    global last_play, mute_until, current_player
    try:
        worth, choice, conf = pick_sfx(text)
    except Exception as e:
        print(f"  (d1 error: {e})")
        return
    print(f"  meme-worthy={worth:.2f}  best={choice}  confidence={conf:.2f}")

    since = time.time() - last_play
    if worth < GATE_THRESHOLD:
        print(f"  -- skipped: meme-worthy {worth:.2f} < {GATE_THRESHOLD}")
    elif conf < MIN_CONFIDENCE:
        print(f"  -- skipped: confidence {conf:.2f} < {MIN_CONFIDENCE}")
    elif since <= COOLDOWN_SEC:
        print(f"  -- skipped: cooldown ({since:.1f}s < {COOLDOWN_SEC}s)")
    else:
        try:
            player = sound_cache[choice]
            player.play()
            current_player = player
            last_play = time.time()
            mute_until = last_play + player.length + 0.4
            print(f"  >> playing {choice}")
        except Exception as e:
            print(f"  (playback error: {e})")

# ---------- whisper ----------
def load_model():
    print("Loading Whisper model on GPU...")
    try:
        m = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
        list(m.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), beam_size=1)[0])
        print("GPU works.")
        return m
    except Exception as e:
        print(f"GPU failed ({e}). Falling back to CPU...")
        m = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=os.cpu_count())
        list(m.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), beam_size=1)[0])
        return m


model = load_model()

print("Warming up d1 (first time can take a minute)...")
try:
    requests.post(OLLAMA_URL, json={
        "model": "d1", "state": "hello", "keep_alive": "60m",
        "questions": {"x": {"type": "noul", "instructions": "Is this a greeting?"}},
    }, timeout=300)
    print("d1 ready.")
except Exception as e:
    print(f"d1 warm-up failed: {e}")


def get_words(audio, beam):
    segments, _ = model.transcribe(
        audio, language=LANGUAGE, beam_size=beam, temperature=0.0,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300),
        condition_on_previous_text=False,
        word_timestamps=True, initial_prompt=PROMPT,
        no_speech_threshold=0.6,
    )
    words = []
    for s in segments:
        if s.no_speech_prob > 0.6:
            continue
        if re.sub(r"[^\w ]", "", s.text.lower()).strip() in HALLUCINATIONS:
            continue
        words += [(w.word.strip(), w.start, w.end) for w in (s.words or []) if w.word.strip()]
    return words


def norm(w):
    return re.sub(r"\W+", "", w.lower())


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if norm(x[0]) != norm(y[0]):
            break
        n += 1
    return n


# ---------- main loop ----------
q = queue.Queue()


def callback(indata, frames, time_info, status):
    q.put(indata[:, 0].copy())


preroll = deque(maxlen=int(PREROLL_SEC / BLOCK_SEC))
buffer = []
speaking = False
silent_blocks = 0
last_partial = 0.0
prev_hyp = []
n_printed = 0
utter_words = []
utter_blocks = 0
blocks_for_end = int(SILENCE_END_SEC / BLOCK_SEC)
max_blocks = int(MAX_UTTERANCE_SEC / BLOCK_SEC)


def reset_utterance():
    global buffer, speaking, silent_blocks, prev_hyp, n_printed, utter_words, utter_blocks
    buffer, speaking, silent_blocks = [], False, 0
    prev_hyp, n_printed, utter_words, utter_blocks = [], 0, [], 0


def emit(words):
    for w, _, _ in words:
        print(w, end=" ", flush=True)
        utter_words.append(w)


print("\nReady! Talk and I'll type live + drop meme sounds (Ctrl+C to stop).\n")
try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=int(SAMPLE_RATE * BLOCK_SEC), callback=callback):
        while True:
            blocks = [q.get()]
            while not q.empty():
                blocks.append(q.get())

            # a sound is playing: only listen for the user talking over it
            if time.time() < mute_until:
                if time.time() - last_play < MIN_PLAY_SEC:
                    bargein_count = 0          # too early to interrupt
                    preroll.clear()
                    continue
                trigger = None
                for i, block in enumerate(blocks):
                    if np.sqrt(np.mean(block ** 2)) > BARGE_IN_THRESHOLD:
                        bargein_count += 1
                    else:
                        bargein_count = 0
                    if bargein_count >= BARGE_IN_BLOCKS:
                        trigger = i
                        break
                if trigger is None:
                    preroll.clear()
                    continue
                # user talked over the sound: stop it and start listening
                if current_player:
                    current_player.stop()
                mute_until = 0.0
                bargein_count = 0
                preroll.clear()
                blocks = blocks[max(0, trigger - BARGE_IN_BLOCKS + 1):]
                print("  (sound stopped, listening...)")

            for block in blocks:
                loud = np.sqrt(np.mean(block ** 2)) > SPEECH_THRESHOLD
                if loud:
                    if not speaking:
                        buffer.extend(preroll)
                        preroll.clear()
                    speaking, silent_blocks = True, 0
                    buffer.append(block)
                    utter_blocks += 1
                elif speaking:
                    silent_blocks += 1
                    buffer.append(block)
                    utter_blocks += 1
                else:
                    preroll.append(block)

            if not speaking:
                continue

            audio = np.concatenate(buffer)
            ended = silent_blocks >= blocks_for_end or utter_blocks >= max_blocks

            if ended:
                if len(audio) > SAMPLE_RATE * 0.3:
                    words = get_words(audio, beam=5)      # accurate final pass
                    emit(words[n_printed:])
                print()
                text = " ".join(utter_words).strip()
                reset_utterance()
                if text:
                    handle(text)
                print()
            elif len(audio) > SAMPLE_RATE * 0.5 and time.time() - last_partial >= PARTIAL_EVERY_SEC:
                hyp = get_words(audio, beam=1)
                last_partial = time.time()
                n_agree = common_prefix_len(prev_hyp, hyp)
                if n_agree > n_printed:
                    emit(hyp[n_printed:n_agree])
                    n_printed = n_agree
                prev_hyp = hyp

                if len(audio) > SAMPLE_RATE * TRIM_AFTER_SEC and n_printed > 0:
                    cut = int(hyp[n_printed - 1][2] * SAMPLE_RATE)
                    buffer = [audio[cut:]]
                    prev_hyp, n_printed = [], 0
except KeyboardInterrupt:
    print("\nStopped.")