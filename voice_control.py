import threading
import time
from typing import Callable

HAVE_VOICE = False
try:
    import sounddevice as sd
    from vosk import Model, KaldiRecognizer
    HAVE_VOICE = True
except Exception:
    HAVE_VOICE = False


class VoiceThread:
    # Background voice command thread.

    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not HAVE_VOICE:
            print("[Voice] Voice support not available; running in no-op mode.")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self):
        # TODO: Implement real recognition using Vosk/ sounddevice.
        print("[Voice] _run not implemented; emitting no commands.")
        while not self._stop.is_set():
            time.sleep(0.1)
