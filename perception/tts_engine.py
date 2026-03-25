import subprocess
import time
import threading


class TTSEngine:
    def __init__(self, repeat_interval=2.0, min_change_interval=0.6, poll_interval=0.1):
        self.repeat_interval = repeat_interval
        self.min_change_interval = min_change_interval
        self.poll_interval = poll_interval

        self.current_message = None
        self.last_spoken_message = None
        self.last_spoken_time = 0

        self.running = True
        self.lock = threading.Lock()
        self.current_proc = None

        self.worker = threading.Thread(target=self._speech_loop, daemon=True)
        self.worker.start()

    def _speak_windows(self, text):
        text = text.replace("'", "''")
        cmd = (
            "Add-Type -AssemblyName System.Speech;"
            "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$speak.Rate = 1;"
            f"$speak.Speak('{text}');"
        )

        self.current_proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        self.current_proc.wait()

    def _speech_loop(self):
        while self.running:
            with self.lock:
                message = self.current_message

            if message is None:
                time.sleep(self.poll_interval)
                continue

            now = time.time()
            same_message_repeat = (
                message == self.last_spoken_message
                and (now - self.last_spoken_time) >= self.repeat_interval
            )
            changed_message_ready = (
                message != self.last_spoken_message
                and (now - self.last_spoken_time) >= self.min_change_interval
            )
            should_speak = same_message_repeat or changed_message_ready

            if should_speak:
                try:
                    if self.current_proc is not None and self.current_proc.poll() is None:
                        self.current_proc.terminate()

                    self._speak_windows(message)
                    self.last_spoken_message = message
                    self.last_spoken_time = time.time()
                except Exception as e:
                    print("TTS error:", e)

            time.sleep(self.poll_interval)

    def speak(self, message):
        with self.lock:
            self.current_message = message

    def clear(self):
        with self.lock:
            self.current_message = None

    def stop(self):
        self.running = False
        try:
            if self.current_proc is not None and self.current_proc.poll() is None:
                self.current_proc.terminate()
        except Exception:
            pass