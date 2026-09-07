# -*- coding: utf-8 -*-
"""
Maya AI - Offline Bengali Live Voice Chat
-----------------------------------------
Features:
- Local Qwen AI
- Bengali text chat
- Offline Bengali voice input with faster-whisper
- Offline Bengali voice output with Piper
- Noto Sans Bengali support
- No cloud API / no Internet required after models are installed
"""

from pathlib import Path
import re
import threading
import wave
import time

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput

from llama_cpp import Llama

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from piper import PiperVoice, SynthesisConfig
except ImportError:
    PiperVoice = None
    SynthesisConfig = None

try:
    import winsound
except ImportError:
    winsound = None


# ============================================================
# SETTINGS / PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

AI_MODEL = BASE / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
WHISPER_MODEL = BASE / "whisper-small"
PIPER_MODEL = BASE / "voice_model" / "bn_BD-google-medium.onnx"

# Bengali Unicode font support
# ------------------------------------------------------------
# Kivy's default Roboto font often cannot render Bengali.
# Prefer a bundled Noto Sans Bengali font, then check common
# Windows font locations. We also scan the local fonts folder
# so renamed/case-varied .ttf files can still be detected.
# ------------------------------------------------------------

FONT_CANDIDATES = [
    BASE / "fonts" / "NotoSansBengali-Regular.ttf",
    BASE / "fonts" / "NotoSansBengali-Regular.TTF",
    BASE / "NotoSansBengali.ttf",
    BASE / "NotoSansBengali.TTF",
    BASE / "NotoBengali-Regular.ttf",
    BASE / "NotoBengali-Regular.TTF",
    Path(r"C:\Windows\Fonts\NotoSansBengali-Regular.ttf"),
    Path(r"C:\Windows\Fonts\NotoSansBengali-Regular.TTF"),
    Path(r"C:\Windows\Fonts\NotoBengali-Regular.ttf"),
    Path(r"C:\Windows\Fonts\NotoBengali-Regular.TTF"),
]

def find_bengali_font():
    """Return the best available Bengali-capable font path."""
    for path in FONT_CANDIDATES:
        try:
            if path.is_file():
                return str(path)
        except OSError:
            pass

    # Scan bundled fonts folder for any likely Bengali font.
    fonts_dir = BASE / "fonts"
    if fonts_dir.is_dir():
        preferred = []
        for path in fonts_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".ttf", ".otf"}:
                name = path.name.lower()
                if "bengali" in name or "bangla" in name or "noto" in name:
                    preferred.append(path)

        if preferred:
            return str(sorted(preferred, key=lambda p: p.name.lower())[0])

    return None

FONT = find_bengali_font()

if FONT is None:
    # Keep the application startable, but make the problem obvious.
    FONT = "Roboto"
    BENGALI_FONT_WARNING = True
else:
    BENGALI_FONT_WARNING = False


SAMPLE_RATE = 16000

# Each voice capture is short so the conversation feels responsive.
RECORD_SECONDS = 4

WHISPER_COMPUTE = "int8"
WHISPER_THREADS = 4

PIPER_SPEAKER_ID = 0

Window.title = "Maya AI - Offline Bengali Voice"
Window.size = (900, 650)


# ============================================================
# MAYA SYSTEM PROMPT
# ============================================================

SYSTEM = """তুমি Maya AI, একজন বন্ধুসুলভ বাংলা AI assistant।

নিয়ম:
১. ব্যবহারকারী বাংলায় প্রশ্ন করলে বাংলায় উত্তর দেবে।
২. অপ্রয়োজনীয় ইংরেজি ব্যবহার করবে না।
৩. ব্যবহারকারী ইংরেজিতে প্রশ্ন করলে ইংরেজিতে উত্তর দিতে পারো।
৪. প্রয়োজন হলে ধাপে ধাপে বুঝিয়ে দেবে।
৫. সংক্ষিপ্ত, পরিষ্কার ও উপকারী উত্তর দেবে।
৬. মিথ্যা তথ্য বানাবে না।
৭. তুমি সম্পূর্ণ local offline AI।
৮. কোনো online API বা Internet ব্যবহার করবে না।
"""


def has_bengali(text):
    return bool(re.search(r"[\u0980-\u09FF]", text or ""))


def safe_bengali_text(text):
    """
    Keep Bengali Unicode intact and remove accidental NUL characters.
    This does not transliterate or replace Bengali characters.
    """
    if text is None:
        return ""
    return str(text).replace("\x00", "").strip()


# ============================================================
# CHAT BUBBLE
# ============================================================

class Bubble(Label):

    def __init__(self, text, user=False, **kwargs):
        super().__init__(**kwargs)

        self.text = safe_bengali_text(text)
        self.font_name = FONT
        self.font_size = 20
        self.halign = "left"
        self.valign = "top"
        self.padding = (16, 12)
        self.size_hint_y = None
        self.markup = False

        with self.canvas.before:
            if user:
                Color(0.08, 0.30, 0.55, 1)
            else:
                Color(0.13, 0.13, 0.13, 1)

            self.bg = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[16]
            )

        self.bind(width=self.wrap_text)
        self.bind(texture_size=self.resize)
        self.bind(pos=self.sync, size=self.sync)

    def wrap_text(self, *_):
        self.text_size = (
            max(100, self.width - 32),
            None
        )

    def resize(self, *_):
        self.height = self.texture_size[1] + 24

    def sync(self, *_):
        self.bg.pos = self.pos
        self.bg.size = self.size


# ============================================================
# MAYA APP
# ============================================================

class MayaAI(App):

    def build(self):

        # ----------------------------------------------------
        # Check Qwen model
        # ----------------------------------------------------

        if not AI_MODEL.exists():
            raise FileNotFoundError(
                "Qwen GGUF model পাওয়া যায়নি:\n\n"
                + str(AI_MODEL)
            )

        # ----------------------------------------------------
        # Load local Qwen
        # ----------------------------------------------------

        self.llm = Llama(
            model_path=str(AI_MODEL),
            n_ctx=2048,
            verbose=False
        )

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM
            }
        ]

        self.whisper = None
        self.piper = None

        self.recording = False
        self.processing_voice = False
        self.tts_lock = threading.Lock()

        # ----------------------------------------------------
        # Main UI
        # ----------------------------------------------------

        root = BoxLayout(
            orientation="vertical",
            padding=12,
            spacing=10
        )

        title = Label(
            text="Maya AI",
            font_name=FONT,
            font_size=30,
            size_hint_y=None,
            height=55
        )

        root.add_widget(title)

        self.status = Label(
            text="OFFLINE AI STARTING...",
            font_name=FONT,
            font_size=15,
            size_hint_y=None,
            height=30
        )

        root.add_widget(self.status)

        # ----------------------------------------------------
        # Chat area
        # ----------------------------------------------------

        self.scroll = ScrollView(
            do_scroll_x=False,
            bar_width=8
        )

        self.chat = BoxLayout(
            orientation="vertical",
            spacing=10,
            padding=5,
            size_hint_y=None
        )

        self.chat.bind(
            minimum_height=self.chat.setter("height")
        )

        self.scroll.add_widget(self.chat)
        root.add_widget(self.scroll)

        self.add_maya(
            "আমি Maya AI।\n"
            "বাংলায় লিখুন অথবা 🎤 VOICE চাপুন।\n"
            "AI, Voice Input এবং Voice Output সব local/offline।"
        )

        # ----------------------------------------------------
        # Bottom input area
        # ----------------------------------------------------

        bottom = BoxLayout(
            size_hint_y=None,
            height=62,
            spacing=7
        )

        self.input = TextInput(
            hint_text="আপনার প্রশ্ন লিখুন...",
            font_name=FONT,
            font_size=19,
            multiline=False,
            padding=(12, 15)
        )

        self.input.bind(
            on_text_validate=self.send
        )

        bottom.add_widget(self.input)

        self.voice_button = Button(
            text="🎤 VOICE",
            font_name=FONT,
            font_size=16,
            size_hint_x=None,
            width=130
        )

        self.voice_button.bind(
            on_press=self.voice_toggle
        )

        bottom.add_widget(self.voice_button)

        self.send_button = Button(
            text="SEND",
            font_name=FONT,
            font_size=18,
            size_hint_x=None,
            width=100
        )

        self.send_button.bind(
            on_press=self.send
        )

        bottom.add_widget(self.send_button)

        root.add_widget(bottom)

        # ----------------------------------------------------
        # Bengali font diagnostic
        # ----------------------------------------------------
        if BENGALI_FONT_WARNING:
            self.set_status("বাংলা Font পাওয়া যায়নি — fonts folder-এ NotoSansBengali-Regular.ttf দিন")
            self.add_maya(
                "বাংলা লেখা সঠিকভাবে দেখাতে NotoSansBengali-Regular.ttf "
                "ফাইলটি MayaAI\\fonts\\ ফোল্ডারে রাখুন।"
            )

        # ----------------------------------------------------
        # Load voice models in background
        # ----------------------------------------------------

        threading.Thread(
            target=self.load_voice_models,
            daemon=True
        ).start()

        return root

    # ========================================================
    # STATUS
    # ========================================================

    def set_status(self, text):
        self.status.text = safe_bengali_text(text)

    def set_status_threadsafe(self, text):
        Clock.schedule_once(
            lambda dt, value=text:
                self.set_status(value)
        )

    # ========================================================
    # LOAD VOICE MODELS
    # ========================================================

    def load_voice_models(self):

        errors = []

        # ----------------------------------------------------
        # Whisper
        # ----------------------------------------------------

        if WhisperModel is None:

            errors.append(
                "faster-whisper নেই। CMD-তে চালান:\n"
                "py -m pip install faster-whisper"
            )

        elif not WHISPER_MODEL.exists():

            errors.append(
                "whisper-small folder নেই:\n"
                + str(WHISPER_MODEL)
            )

        else:

            try:

                self.whisper = WhisperModel(
                    str(WHISPER_MODEL),
                    device="cpu",
                    compute_type=WHISPER_COMPUTE,
                    cpu_threads=WHISPER_THREADS,
                    num_workers=1
                )

            except Exception as e:

                errors.append(
                    "Whisper load error: " + str(e)
                )

        # ----------------------------------------------------
        # Piper
        # ----------------------------------------------------

        if PiperVoice is None:

            errors.append(
                "Piper নেই। CMD-তে চালান:\n"
                "py -m pip install piper-tts"
            )

        elif not PIPER_MODEL.exists():

            errors.append(
                "Bengali Piper voice নেই:\n"
                + str(PIPER_MODEL)
            )

        else:

            try:

                self.piper = PiperVoice.load(
                    str(PIPER_MODEL)
                )

            except Exception as e:

                errors.append(
                    "Piper load error: " + str(e)
                )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        if not errors:

            Clock.schedule_once(
                lambda dt:
                    self.set_status(
                        "OFFLINE AI + BANGLA VOICE READY"
                    )
            )

        else:

            msg = "\n".join(errors)

            Clock.schedule_once(
                lambda dt, value=msg:
                    self.setup_warning(value)
            )

    def setup_warning(self, msg):

        # Show the setup problem without crashing the Kivy event loop.
        self.set_status("OFFLINE AI READY - VOICE SETUP WARNING")

        self.add_maya(
            "AI চালু আছে। Voice-এর কিছু অংশ setup করা বাকি:\n\n"
            + safe_bengali_text(msg)
        )


    # ========================================================
    # CHAT UI
    # ========================================================

    def add_user(self, text):

        text = safe_bengali_text(text)

        self.chat.add_widget(
            Bubble(
                "আপনি: " + text,
                user=True
            )
        )

        self.scroll_bottom()

    def add_maya(self, text):

        text = safe_bengali_text(text)

        self.chat.add_widget(
            Bubble(
                "Maya: " + text,
                user=False
            )
        )

        self.scroll_bottom()

    def scroll_bottom(self):

        Clock.schedule_once(
            lambda dt:
                setattr(
                    self.scroll,
                    "scroll_y",
                    0
                ),
            0.05
        )

    # ========================================================
    # SEND MESSAGE
    # ========================================================

    def send(self, *_):

        q = self.input.text.strip()

        if not q:
            return

        self.add_user(q)

        self.input.text = ""

        self.input.disabled = True
        self.send_button.disabled = True
        self.voice_button.disabled = True

        self.set_status(
            "MAYA THINKING..."
        )

        threading.Thread(
            target=self.answer,
            args=(q,),
            daemon=True
        ).start()

    # ========================================================
    # QWEN ANSWER
    # ========================================================

    def answer(self, q):

        try:

            self.messages.append(
                {
                    "role": "user",
                    "content": q
                }
            )

            result = self.llm.create_chat_completion(
                messages=self.messages,
                max_tokens=512,
                temperature=0.7
            )

            answer = (
                result["choices"][0]["message"]["content"]
                .strip()
            )
            answer = safe_bengali_text(answer)

            self.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            answer = (
                "দুঃখিত, উত্তর তৈরি করতে সমস্যা হয়েছে।\n"
                + str(e)
            )

        Clock.schedule_once(
            lambda dt, value=answer:
                self.answer_done(value)
        )

    def answer_done(self, answer):

        self.add_maya(answer)

        self.input.disabled = False
        self.send_button.disabled = False
        self.voice_button.disabled = False

        self.input.focus = True

        self.set_status(
            "OFFLINE AI + BANGLA VOICE READY"
        )

        # Speak Bengali answer locally.
        if has_bengali(answer):

            threading.Thread(
                target=self.speak_bengali,
                args=(answer,),
                daemon=True
            ).start()

    # ========================================================
    # VOICE INPUT
    # ========================================================

    def voice_toggle(self, *_):

        # Stop current recording
        if self.recording:

            self.stop_recording()
            return

        # Whisper not ready
        if self.whisper is None:

            self.add_maya(
                "Offline Bengali Whisper model এখনও ready হয়নি।\n"
                "whisper-small folder পরীক্ষা করুন।"
            )

            return

        # Sounddevice not ready
        if sd is None:

            self.add_maya(
                "sounddevice নেই। CMD-তে চালান:\n"
                "py -m pip install sounddevice numpy"
            )

            return

        self.recording = True
        self.processing_voice = False

        self.voice_button.text = "⏹ STOP"

        self.input.disabled = True
        self.send_button.disabled = True

        self.set_status(
            "🎤 LISTENING... বাংলায় কথা বলুন"
        )

        threading.Thread(
            target=self.record_and_transcribe,
            daemon=True
        ).start()

    def stop_recording(self):

        self.recording = False

        try:
            sd.stop()
        except Exception:
            pass

        self.voice_button.text = "🎤 VOICE"

        self.input.disabled = False
        self.send_button.disabled = False

        self.set_status(
            "TRANSCRIPTION STOPPED"
        )

    # ========================================================
    # RECORD AUDIO + WHISPER
    # ========================================================

    def record_and_transcribe(self):

        try:

            audio = sd.rec(
                int(RECORD_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )

            # Allow STOP button to interrupt recording.
            for _ in range(RECORD_SECONDS * 10):

                if not self.recording:

                    try:
                        sd.stop()
                    except Exception:
                        pass

                    return

                time.sleep(0.1)

            sd.wait()

            if not self.recording:
                return

            self.processing_voice = True

            self.set_status_threadsafe(
                "WHISPER → বাংলা লেখা তৈরি হচ্ছে..."
            )

            segments, info = self.whisper.transcribe(
                audio[:, 0],
                language="bn",
                task="transcribe",
                beam_size=3,
                vad_filter=True,
                condition_on_previous_text=False
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
            ).strip()

            self.recording = False
            self.processing_voice = False

            Clock.schedule_once(
                lambda dt, value=text:
                    self.voice_done(value)
            )

        except Exception as e:

            self.recording = False
            self.processing_voice = False

            Clock.schedule_once(
                lambda dt, value=str(e):
                    self.voice_error(value)
            )

    def voice_done(self, text):

        self.voice_button.text = "🎤 VOICE"

        self.input.disabled = False
        self.send_button.disabled = False

        if not text:

            self.set_status(
                "OFFLINE AI + BANGLA VOICE READY"
            )

            self.add_maya(
                "কোনো কথা শনাক্ত হয়নি। আবার VOICE চাপুন।"
            )

            return

        # Show recognized Bengali text.
        text = safe_bengali_text(text)
        self.input.text = text

        self.set_status(
            "BANGLA VOICE RECOGNIZED"
        )

        # Automatically send to local Qwen.
        self.send()

    def voice_error(self, error):

        self.voice_button.text = "🎤 VOICE"

        self.input.disabled = False
        self.send_button.disabled = False

        self.set_status(
            "VOICE ERROR"
        )

        self.add_maya(
            "Voice error:\n" + error
        )

    # ========================================================
    # PIPER BANGLA VOICE OUTPUT
    # ========================================================

    def speak_bengali(self, text):

        if self.piper is None:
            return

        if winsound is None:
            return

        with self.tts_lock:

            wav_path = BASE / "_maya_tts.wav"

            try:

                if SynthesisConfig is not None:

                    config = SynthesisConfig(
                        speaker_id=PIPER_SPEAKER_ID,
                        length_scale=1.0,
                        noise_scale=0.667,
                        noise_w_scale=0.8
                    )

                    with wave.open(
                        str(wav_path),
                        "wb"
                    ) as wav_file:

                        self.piper.synthesize_wav(
                            text,
                            wav_file,
                            syn_config=config
                        )

                else:

                    with wave.open(
                        str(wav_path),
                        "wb"
                    ) as wav_file:

                        self.piper.synthesize_wav(
                            text,
                            wav_file
                        )

                winsound.PlaySound(
                    str(wav_path),
                    winsound.SND_FILENAME
                )

            except Exception as e:

                Clock.schedule_once(
                    lambda dt, value=str(e):
                        self.add_maya(
                            "বাংলা voice output error:\n"
                            + value
                        )
                )

            finally:

                try:

                    if wav_path.exists():
                        wav_path.unlink()

                except Exception:
                    pass


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    MayaAI().run()
