"""Tkinter UI for simulated voice-command music control.

Layout targets a simple music player structure:
[Placeholder Image]
Song Name
<time/duration bar>
<previous> <stop/play> <next>
[List of songs]

Hands-free mode:
- Automatically starts background voice listening on launch (if available).
- Processes spoken commands without pressing any button.
"""

from __future__ import annotations

import argparse
import queue
import re
import threading
import time
import wave
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

try:
    from nlp_metadata import interpret_command
except ImportError:
    from .nlp_metadata import interpret_command

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import pygame
except ImportError:
    pygame = None

try:
    from mutagen import File as MutagenFile
except ImportError:
    MutagenFile = None


SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a"}


@dataclass(slots=True)
class TrackInfo:
    title: str
    artist: str
    album: str
    duration_seconds: int
    file_path: Path | None = None


class AudioBackend:
    """Thin wrapper around pygame music playback."""

    def __init__(self) -> None:
        # `available` indicates whether real playback can be used at runtime.
        self.available = False
        self.error_message: str | None = None

        if pygame is None:
            self.error_message = "pygame is not installed"
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self.available = True
        except Exception as exc:
            self.error_message = f"pygame mixer init failed: {exc}"

    def play(self, file_path: Path) -> bool:
        if not self.available:
            return False
        try:
            pygame.mixer.music.load(str(file_path))
            pygame.mixer.music.play()
            return True
        except Exception as exc:
            self.error_message = f"audio playback failed: {exc}"
            return False

    def stop(self) -> None:
        if self.available:
            pygame.mixer.music.stop()


def format_transcript_for_display(transcript: str) -> str:
    """Uppercase only the first letter for display."""

    clean = transcript.strip()
    if not clean:
        return clean
    return clean[0].upper() + clean[1:]


def get_audio_duration_seconds(file_path: Path) -> int:
    """Read duration from real audio files when possible."""

    # Prefer mutagen first because it supports multiple file formats.
    if MutagenFile is not None:
        try:
            audio_file = MutagenFile(str(file_path))
            if audio_file is not None and getattr(audio_file, "info", None) is not None:
                length = getattr(audio_file.info, "length", None)
                if length is not None:
                    return max(1, int(round(float(length))))
        except Exception:
            pass

    # WAV fallback works even when mutagen cannot parse the file.
    if file_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(file_path), "rb") as wav_file:
                frame_count = wav_file.getnframes()
                frame_rate = wav_file.getframerate()
                return max(1, int(round(frame_count / float(frame_rate))))
        except Exception:
            pass

    # Final fallback keeps the UI usable even if metadata probing fails.
    return 180


def parse_track_name(file_path: Path) -> TrackInfo:
    """Build display metadata from a music filename.

    Best-case filename format:
    Artist - Song Title.mp3
    """

    stem = file_path.stem.strip()
    duration_seconds = get_audio_duration_seconds(file_path)

    if " - " in stem:
        artist, title = stem.split(" - ", maxsplit=1)
        return TrackInfo(
            title=title.strip() or stem,
            artist=artist.strip() or "Unknown Artist",
            album="Local Music",
            duration_seconds=duration_seconds,
            file_path=file_path,
        )

    return TrackInfo(
        title=stem or "Unknown Track",
        artist="Unknown Artist",
        album="Local Music",
        duration_seconds=duration_seconds,
        file_path=file_path,
    )


def resolve_music_dir(project_root: Path, provided_dir: Path | None) -> Path:
    """Resolve the music folder, checking likely project/workspace locations."""

    candidates: list[Path] = []
    if provided_dir is not None:
        candidates.append(provided_dir)

    candidates.extend(
        [
            project_root / "data" / "music",
            project_root / "music",
            project_root.parent / "music",
            project_root.parent / "data" / "music",
        ]
    )

    # Return the first existing directory; otherwise keep the top priority path
    # so user messages can show where music files should be placed.
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return candidates[0]


def load_tracks_from_music_dir(music_dir: Path) -> list[TrackInfo]:
    """Load playlist entries directly from the local music folder."""

    if not music_dir.exists():
        return []

    tracks: list[TrackInfo] = []
    for file_path in sorted(music_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            tracks.append(parse_track_name(file_path))
    return tracks


class SimulatedPlayer:
    """Small player state machine used by the UI."""

    def __init__(self, tracks: list[TrackInfo], audio_backend: AudioBackend) -> None:
        self.playlist = tracks or [TrackInfo("No songs found", "Add files to data/music", "Local Music", 180, None)]
        self.audio_backend = audio_backend
        self.current_index = 0
        self.state = "stopped"
        self.position_seconds = 0

    @property
    def current_track(self) -> TrackInfo:
        return self.playlist[self.current_index]

    def play(self) -> str:
        # Guard when no real tracks are present.
        if not self.current_track.file_path:
            return "No real songs available in the music folder yet."

        if not self.audio_backend.play(self.current_track.file_path):
            return self.audio_backend.error_message or "Audio playback is unavailable."

        self.position_seconds = 0
        self.state = "playing"
        return f"Playing: {self.current_track.title}"

    def stop(self) -> str:
        self.audio_backend.stop()
        self.state = "stopped"
        self.position_seconds = 0
        return "Stopped playback"

    def next(self) -> str:
        if len(self.playlist) == 1 and self.current_track.file_path is None:
            return "No real songs available in the music folder yet."
        self.current_index = (self.current_index + 1) % len(self.playlist)
        return self.play()

    def previous(self) -> str:
        if len(self.playlist) == 1 and self.current_track.file_path is None:
            return "No real songs available in the music folder yet."
        self.current_index = (self.current_index - 1) % len(self.playlist)
        return self.play()

    def set_track_by_title(self, requested_title: str) -> bool:
        # Basic exact/substring match keeps matching behavior predictable.
        wanted = requested_title.strip().lower()
        for index, track in enumerate(self.playlist):
            if track.file_path and (track.title.lower() == wanted or wanted in track.title.lower()):
                self.current_index = index
                self.position_seconds = 0
                return True
        return False

    def tick(self) -> bool:
        """Advance playback by one simulated second."""

        if self.state != "playing" or not self.current_track.file_path:
            return False

        self.position_seconds += 1
        if self.position_seconds >= self.current_track.duration_seconds:
            self.next()
            return True
        return False


class PlayerUI:
    """Tkinter application with requested layout and hands-free voice control."""

    def __init__(self, root: tk.Tk, music_dir: Path | None = None) -> None:
        # Root window sizing favors demo readability on common laptop screens.
        self.root = root
        self.root.title("Voice Command Music Simulator")
        self.root.geometry("850x700")

        self.project_root = Path(__file__).resolve().parent.parent
        self.music_dir = resolve_music_dir(self.project_root, music_dir)
        # Playlist is built from local files only (no hardcoded demo songs).
        tracks = load_tracks_from_music_dir(self.music_dir)
        self.loaded_from_music_dir = bool(tracks)

        self.audio_backend = AudioBackend()
        self.player = SimulatedPlayer(tracks, self.audio_backend)
        # Worker thread posts transcripts/errors into this queue for safe
        # handling on the Tkinter main thread.
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()

        self.voice_running = False
        self.last_voice_transcript = ""
        self.last_voice_time = 0.0
        self.voice_cooldown_seconds = 1.2
        self.toast_hide_job: str | None = None

        self._build_ui()
        self._refresh_ui()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(1000, self._timer_tick)
        self.root.after(150, self._poll_worker_queue)
        self._start_hands_free_voice_loop()

    def _build_ui(self) -> None:
        # Main container for all visual sections.
        self.wrapper = ttk.Frame(self.root, padding=12)
        self.wrapper.pack(fill="both", expand=True)

        header = ttk.Label(self.wrapper, text="TUGTUGAN NI RANDEL YUMUL", font=("Segoe UI", 16, "bold"))
        header.pack(anchor="center", pady=(0, 12))

        self.song_title_var = tk.StringVar(value="-")
        self.song_subtitle_var = tk.StringVar(value="-")

        # Simple cover art placeholder with dynamic song title text.
        self.cover_canvas = tk.Canvas(self.wrapper, width=260, height=260, bg="#222222", highlightthickness=0)
        self.cover_canvas.pack(pady=(0, 8))
        self.cover_canvas.create_rectangle(20, 20, 240, 240, fill="#444444", outline="#777777")
        self.cover_text_id = self.cover_canvas.create_text(130, 130, text="", fill="white", font=("Segoe UI", 12, "bold"), width=190, justify="center")

        ttk.Label(self.wrapper, textvariable=self.song_title_var, font=("Segoe UI", 14, "bold")).pack()
        ttk.Label(self.wrapper, textvariable=self.song_subtitle_var, font=("Segoe UI", 10)).pack(pady=(0, 8))

        time_frame = ttk.Frame(self.wrapper)
        time_frame.pack(fill="x", padx=40)

        self.time_var = tk.StringVar(value="00:00 / 00:00")
        ttk.Label(time_frame, textvariable=self.time_var).pack(anchor="center")

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(time_frame, orient="horizontal", mode="determinate", variable=self.progress_var)
        self.progress_bar.pack(fill="x", pady=(4, 12))

        # Transport controls mirror common player interactions.
        controls = ttk.Frame(self.wrapper)
        controls.pack(pady=(0, 12))

        ttk.Button(controls, text="Previous", command=self._on_previous).grid(row=0, column=0, padx=6)
        self.play_stop_button = ttk.Button(controls, text="Play", command=self._on_play_stop)
        self.play_stop_button.grid(row=0, column=1, padx=6)
        ttk.Button(controls, text="Next", command=self._on_next).grid(row=0, column=2, padx=6)

        # Playlist listbox allows manual song selection.
        list_frame = ttk.LabelFrame(self.wrapper, text="List of Songs", padding=10)
        list_frame.pack(fill="both", expand=True)

        self.song_list = tk.Listbox(list_frame, height=12)
        self.song_list.pack(fill="both", expand=True)
        self.song_list.bind("<<ListboxSelect>>", self._on_list_select)

        for track in self.player.playlist:
            self.song_list.insert("end", f"{track.title} - {track.artist}")

        log_frame = ttk.LabelFrame(self.wrapper, text="Event Log", padding=8)
        log_frame.pack(fill="both", expand=True, pady=(10, 0))

        self.log = tk.Text(log_frame, height=8, wrap="word")
        self.log.pack(fill="both", expand=True)
        self._append_log("UI ready. Speak commands directly: play, stop, next, previous.")
        if self.loaded_from_music_dir:
            self._append_log(f"Loaded {len(self.player.playlist)} track(s) from {self.music_dir}")
        else:
            self._append_log(f"No compatible music files found in {self.music_dir}")
        if not self.audio_backend.available:
            self._append_log(self.audio_backend.error_message or "Audio playback unavailable.")

        # Floating in-window toast for recognized voice command feedback.
        self.toast_label = tk.Label(
            self.wrapper,
            text="",
            bg="#111111",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=18,
            pady=10,
            relief="solid",
            bd=1,
        )
        self.toast_label.place_forget()

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _show_command_toast(self, transcript: str) -> None:
        # Auto-hide timer avoids stale overlays while keeping commands visible.
        self.toast_label.configure(text=f"Voice Command: {format_transcript_for_display(transcript)}")
        self.toast_label.place(relx=0.5, rely=0.38, anchor="center")

        if self.toast_hide_job is not None:
            self.root.after_cancel(self.toast_hide_job)
        self.toast_hide_job = self.root.after(2200, self._hide_command_toast)

    def _hide_command_toast(self) -> None:
        self.toast_label.place_forget()
        self.toast_hide_job = None

    def _seconds_to_mmss(self, seconds: int) -> str:
        minutes = seconds // 60
        remaining = seconds % 60
        return f"{minutes:02d}:{remaining:02d}"

    def _refresh_ui(self) -> None:
        # Sync all UI fields from current player state.
        track = self.player.current_track
        self.song_title_var.set(track.title)
        self.song_subtitle_var.set(f"{track.artist} | {track.album} | {self.player.state.upper()}")
        self.cover_canvas.itemconfig(self.cover_text_id, text=track.title)
        self.cover_canvas.itemconfig(self.cover_text_id, text=track.title)

        total = track.duration_seconds
        current = min(self.player.position_seconds, total)
        self.time_var.set(f"{self._seconds_to_mmss(current)} / {self._seconds_to_mmss(total)}")
        self.progress_var.set((current / total) * 100 if total > 0 else 0)
        self.play_stop_button.configure(text="Stop" if self.player.state == "playing" else "Play")

        self.song_list.selection_clear(0, "end")
        self.song_list.selection_set(self.player.current_index)
        self.song_list.see(self.player.current_index)

    def _timer_tick(self) -> None:
        # Heartbeat: advance playback clock and refresh widgets every second.
        changed = self.player.tick()
        if changed:
            self._append_log(f"Auto-next: {self.player.current_track.title}")
        self._refresh_ui()
        self.root.after(1000, self._timer_tick)

    def _on_close(self) -> None:
        self.voice_running = False
        self.audio_backend.stop()
        self.root.destroy()

    def _on_previous(self) -> None:
        self._append_log(self.player.previous())
        self._refresh_ui()

    def _on_play_stop(self) -> None:
        message = self.player.stop() if self.player.state == "playing" else self.player.play()
        self._append_log(message)
        self._refresh_ui()

    def _on_next(self) -> None:
        self._append_log(self.player.next())
        self._refresh_ui()

    def _on_list_select(self, event: tk.Event) -> None:
        selection = self.song_list.curselection()
        if not selection:
            return
        self.player.current_index = selection[0]
        self.player.position_seconds = 0
        message = self.player.play()
        self._append_log(f"Selected from list: {self.player.current_track.title}")
        self._append_log(message)
        self._refresh_ui()

    def _start_hands_free_voice_loop(self) -> None:
        # Voice worker starts once and continuously listens in background.
        if sr is None:
            self._append_log("Hands-free voice disabled: SpeechRecognition not installed.")
            return

        if self.voice_running:
            return

        self.voice_running = True
        self._append_log("Hands-free voice enabled.")
        worker = threading.Thread(target=self._continuous_voice_worker, daemon=True)
        worker.start()

    def _continuous_voice_worker(self) -> None:
        # Runs outside Tkinter thread to avoid UI freezes during mic/API waits.
        recognizer = sr.Recognizer()

        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.6)
                while self.voice_running:
                    try:
                        audio = recognizer.listen(source, timeout=2, phrase_time_limit=5)
                    except sr.WaitTimeoutError:
                        continue

                    try:
                        transcript = recognizer.recognize_google(audio)
                        self.queue.put(("transcript", transcript))
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as exc:
                        self.queue.put(("error", f"Speech API error: {exc}"))
                        time.sleep(2)

        except Exception as exc:
            self.queue.put(("error", f"Microphone error: {exc}"))

    def _poll_worker_queue(self) -> None:
        # Drain worker messages on main thread, then schedule next poll.
        try:
            while True:
                kind, value = self.queue.get_nowait()

                if kind == "transcript":
                    transcript = value.strip()
                    if not transcript:
                        continue

                    now = time.time()
                    # Duplicate cooldown prevents the same command from firing
                    # repeatedly when ASR returns near-identical transcripts.
                    duplicate = transcript.lower() == self.last_voice_transcript.lower()
                    if duplicate and (now - self.last_voice_time) < self.voice_cooldown_seconds:
                        continue

                    self.last_voice_transcript = transcript
                    self.last_voice_time = now

                    display_transcript = format_transcript_for_display(transcript)
                    self._append_log(f"Recognized voice command: {display_transcript}")
                    self._show_command_toast(display_transcript)
                    self._handle_command(transcript)
                else:
                    self._append_log(f"Voice error: {value}")

        except queue.Empty:
            pass

        self.root.after(150, self._poll_worker_queue)

    def _handle_command(self, transcript: str) -> None:
        # Fast-path regex for previous/back variants before NLP intent mapping.
        lowered = transcript.lower().strip()
        display_transcript = format_transcript_for_display(transcript)

        if re.search(r"\b(previous|prev|back)\b", lowered):
            self._append_log(f"Command: {display_transcript}")
            self._append_log(self.player.previous())
            self._refresh_ui()
            return

        # NLP stage resolves play/pause/next/stop and optional song title.
        interpretation = interpret_command(
            transcript,
            [{"title": t.title, "artist": t.artist, "album": t.album} for t in self.player.playlist if t.file_path],
        )

        self._append_log(f"Command: {display_transcript}")

        if interpretation.intent == "play":
            if interpretation.matched_song:
                self.player.set_track_by_title(interpretation.matched_song.title)
            message = self.player.play()
            if interpretation.song_query and not interpretation.matched_song:
                message += f" | No confident song match for '{interpretation.song_query}'"
            self._append_log(message)
        elif interpretation.intent in {"pause", "stop"}:
            self._append_log(self.player.stop())
        elif interpretation.intent == "next":
            self._append_log(self.player.next())
        else:
            self._append_log("Unknown command. Try: play, stop, next, previous.")

        self._refresh_ui()


def parse_args() -> argparse.Namespace:
    """Parse UI runtime options."""

    parser = argparse.ArgumentParser(description="Tkinter music simulation UI with hands-free voice commands")
    parser.add_argument("--music-dir", type=Path, default=None, help="Optional music folder override")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for launching the Tkinter app."""

    args = parse_args()
    root = tk.Tk()
    PlayerUI(root, music_dir=args.music_dir)
    root.mainloop()


if __name__ == "__main__":
    main()

