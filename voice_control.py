import threading
import time
from typing import Callable
import json

HAVE_VOICE = False
try:
    import sounddevice as sd
    from vosk import Model, KaldiRecognizer
    HAVE_VOICE = True
except Exception:
    HAVE_VOICE = False

MODEL_PATH = "/home/alexnz2/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000

GRAMMAR = '["left", "right", "forward", "back", "spin", "emotion on", "emotion off", "stop", "[unk]"]'


class VoiceThread:
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not HAVE_VOICE:
            print("[Voice] Voice not available; running in no-op mode.")
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self):
        print("[Voice] Starting Vosk recognition…")

        model = Model(MODEL_PATH)
        recognizer = KaldiRecognizer(model, SAMPLE_RATE, GRAMMAR)

        def audio_callback(indata, frames, time_info, status):
            if self._stop.is_set():
                raise sd.CallbackStop()

            # in RawInputStream, indata is a low-level buffer; wrap it in bytes
            data = bytes(indata)
            if recognizer.AcceptWaveform(data):
                result_json = recognizer.Result()
                try:
                    result = json.loads(result_json)
                except json.JSONDecodeError:
                    return
                text = result.get("text", "").strip()
                if text:
                    print(f"[Voice] Heard: {text!r}")
                    self.callback(text)
            else:
                # ignore partial results for now
                pass


        try:
            with sd.RawInputStream(samplerate=SAMPLE_RATE,
                                   blocksize=BLOCK_SIZE,
                                   dtype="int16",
                                   channels=1,
                                   callback=audio_callback):
                while not self._stop.is_set():
                    time.sleep(0.1)
        except Exception as e:
            print("[Voice] ERROR:", e)
