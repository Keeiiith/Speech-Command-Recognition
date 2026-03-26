"""Tkinter UI for simulated voice-command music control.

Layout targets a circular mobile music player structure.
"""

from __future__ import annotations

import argparse
import math
import queue
import re
import threading
import time
import wave
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from pathlib import Path

# Try importing PIL for the rotating album art feature
try:
    from PIL import Image, ImageTk, ImageDraw, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from nlp_metadata import interpret_command
except ImportError:
    # Fallback for missing NLP logic in standalone testing
    class DummyInterpretation:
        def __init__(self, intent, song_query, matched_song):
            self.intent = intent
            self.song_query = song_query
            self.matched_song = matched_song
    def interpret_command(t, p): return DummyInterpretation("unknown", None, None)

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

    def pause(self) -> None:
        if self.available:
            pygame.mixer.music.pause()

    def unpause(self) -> None:
        if self.available:
            pygame.mixer.music.unpause()

    def stop(self) -> None:
        if self.available:
            pygame.mixer.music.stop()


def format_transcript_for_display(transcript: str) -> str:
    clean = transcript.strip()
    if not clean:
        return clean
    return clean[0].upper() + clean[1:]


def get_audio_duration_seconds(file_path: Path) -> int:
    if MutagenFile is not None:
        try:
            audio_file = MutagenFile(str(file_path))
            if audio_file is not None and getattr(audio_file, "info", None) is not None:
                length = getattr(audio_file.info, "length", None)
                if length is not None:
                    return max(1, int(round(float(length))))
        except Exception:
            pass

    if file_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(file_path), "rb") as wav_file:
                frame_count = wav_file.getnframes()
                frame_rate = wav_file.getframerate()
                return max(1, int(round(frame_count / float(frame_rate))))
        except Exception:
            pass

    return 180


def parse_track_name(file_path: Path) -> TrackInfo:
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

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return candidates[0]


def load_tracks_from_music_dir(music_dir: Path) -> list[TrackInfo]:
    if not music_dir.exists():
        return []

    tracks: list[TrackInfo] = []
    for file_path in sorted(music_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            tracks.append(parse_track_name(file_path))
    return tracks


class SimulatedPlayer:
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
        if not self.current_track.file_path:
            return "No real songs available in the music folder yet."

        # If currently paused and we haven't changed the track position manually, just unpause
        if self.state == "paused" and self.position_seconds > 0:
            return self.resume()

        if not self.audio_backend.play(self.current_track.file_path):
            return self.audio_backend.error_message or "Audio playback is unavailable."

        self.position_seconds = 0
        self.state = "playing"
        return f"Playing: {self.current_track.title}"

    def pause(self) -> str:
        self.audio_backend.pause()
        self.state = "paused"
        return f"Paused: {self.current_track.title}"

    def resume(self) -> str:
        self.audio_backend.unpause()
        self.state = "playing"
        return f"Resumed: {self.current_track.title}"

    def stop(self) -> str:
        self.audio_backend.stop()
        self.state = "stopped"
        self.position_seconds = 0
        return "Stopped playback"

    def next(self) -> str:
        if len(self.playlist) == 1 and self.current_track.file_path is None:
            return "No real songs available in the music folder yet."
        self.current_index = (self.current_index + 1) % len(self.playlist)
        self.position_seconds = 0 # Force a restart for the new track
        self.state = "stopped" # Reset state to ensure full play triggers
        return self.play()

    def previous(self) -> str:
        if len(self.playlist) == 1 and self.current_track.file_path is None:
            return "No real songs available in the music folder yet."
        self.current_index = (self.current_index - 1) % len(self.playlist)
        self.position_seconds = 0 # Force a restart for the new track
        self.state = "stopped"
        return self.play()

    def set_track_by_title(self, requested_title: str) -> bool:
        wanted = requested_title.strip().lower()
        for index, track in enumerate(self.playlist):
            if track.file_path and (track.title.lower() == wanted or wanted in track.title.lower()):
                self.current_index = index
                self.position_seconds = 0
                self.state = "stopped"
                return True
        return False

    def tick(self) -> bool:
        if self.state != "playing" or not self.current_track.file_path:
            return False

        self.position_seconds += 1
        if self.position_seconds >= self.current_track.duration_seconds:
            self.next()
            return True
        return False


class PlayerUI:
    def __init__(self, root: tk.Tk, music_dir: Path | None = None) -> None:
        self.root = root
        self.root.title("VC Music Player")
        self.root.geometry("400x850")

        self.project_root = Path(__file__).resolve().parent.parent
        self.music_dir = resolve_music_dir(self.project_root, music_dir)
        tracks = load_tracks_from_music_dir(self.music_dir)
        self.loaded_from_music_dir = bool(tracks)

        self.audio_backend = AudioBackend()
        self.player = SimulatedPlayer(tracks, self.audio_backend)
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()

        self.voice_running = False
        self.last_voice_transcript = ""
        self.last_voice_time = 0.0
        self.voice_cooldown_seconds = 1.2
        self.toast_hide_job: str | None = None
        
        self.rotation_angle = 0.0  # For smooth album rotation
        self.last_rendered_index = -1 # Used to detect when we need to rebuild the playlist UI

        self._build_ui()
        self._refresh_ui()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(1000, self._timer_tick)
        self.root.after(50, self._spin_tick) # Fast tick for smooth UI rotation
        self.root.after(150, self._poll_worker_queue)
        self._start_hands_free_voice_loop()

    def _build_ui(self) -> None:
        # Theme Colors
        self.BG_COLOR = "#181A20"      # Deep Dark Grey
        self.BG_DARK = "#121419"       # Darker for bottom section
        self.FG_PRIMARY = "#FFFFFF"    # White text
        self.FG_SECONDARY = "#828795"  # Muted Grey text
        self.ACCENT_CYAN = "#33C5CE"   # Bright Cyan

        self.root.configure(bg=self.BG_COLOR)

        # Configure custom themed scrollbar
        self.style = ttk.Style()
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")
            
        self.style.configure(
            "Custom.Vertical.TScrollbar",
            gripcount=0,
            background="#2A2E38",         # The scrollbar thumb color
            troughcolor=self.BG_DARK,     # The scrollbar track
            bordercolor=self.BG_DARK,
            darkcolor=self.BG_DARK,
            lightcolor=self.BG_DARK,
            arrowcolor=self.ACCENT_CYAN,  # Arrows at top and bottom
            relief="flat"
        )
        self.style.map("Custom.Vertical.TScrollbar", background=[("active", self.ACCENT_CYAN)])

        self.wrapper = tk.Frame(self.root, bg=self.BG_COLOR)
        self.wrapper.pack(fill="both", expand=True)

        # 1. Top Bar
        top_bar = tk.Frame(self.wrapper, bg=self.BG_COLOR, pady=15, padx=20)
        top_bar.pack(fill="x")
        tk.Label(top_bar, text="NOW PLAYING", font=("Segoe UI", 9, "bold"), bg=self.BG_COLOR, fg=self.FG_PRIMARY).pack(side="left", expand=True)

        # 2. Circular Stage setup
        stage_frame = tk.Frame(self.wrapper, bg=self.BG_COLOR)
        stage_frame.pack(fill="x", pady=(10, 0))
        
        self.canvas_size = 300
        self.center_xy = self.canvas_size / 2
        self.progress_radius = 120
        self.art_radius = 100

        self.cover_canvas = tk.Canvas(stage_frame, width=self.canvas_size, height=self.canvas_size, bg=self.BG_COLOR, highlightthickness=0)
        self.cover_canvas.pack(pady=10)

        # Base Image Placeholder setup (Perfect Circular Crop)
        self.original_image = None
        self.tk_image = None
        self.thumb_tk_image = None
        
        if HAS_PIL:
            try:
                # Use forward slash for cross-platform compatibility
                img_path = Path("data/cover/cat.jpg")
                if img_path.exists():
                    self.original_image = Image.open(img_path).convert("RGBA")
                else:
                    self.original_image = Image.new("RGBA", (200, 200), (42, 46, 56, 255))
                    d = ImageDraw.Draw(self.original_image)
                    d.text((45, 95), "Add cover.jpg", fill=(130, 135, 149, 255))
                
                # Resize and create a perfect circular mask
                size = (int(self.art_radius * 2), int(self.art_radius * 2))
                self.original_image = ImageOps.fit(self.original_image, size, centering=(0.5, 0.5))
                
                mask = Image.new('L', size, 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse((0, 0, size[0], size[1]), fill=255)
                self.original_image.putalpha(mask)
                
                # Generate a smaller thumbnail for the playlist items
                thumb_img = self.original_image.resize((44, 44), Image.Resampling.LANCZOS)
                self.thumb_tk_image = ImageTk.PhotoImage(thumb_img)
                
            except Exception:
                pass

        # Layer 1: Album Art Image
        self.album_art_id = self.cover_canvas.create_image(self.center_xy, self.center_xy, image=None, tags="layer1")

        # Layer 3: Center Vinyl Hole
        self.cover_canvas.create_oval(
            self.center_xy - 15, self.center_xy - 15,
            self.center_xy + 15, self.center_xy + 15,
            fill=self.BG_COLOR, outline=self.ACCENT_CYAN, width=4, tags="layer3"
        )

        # Layer 4: Dark Progress Track Background
        self.cover_canvas.create_oval(
            self.center_xy - self.progress_radius, self.center_xy - self.progress_radius,
            self.center_xy + self.progress_radius, self.center_xy + self.progress_radius,
            outline="#2A2E38", width=3, tags="layer4"
        )

        # Layer 6: Progress Dot
        self.progress_dot = self.cover_canvas.create_oval(0, 0, 0, 0, fill=self.ACCENT_CYAN, outline=self.ACCENT_CYAN, tags="layer6")

        # Layer 7: Time Texts on the canvas
        self.time_current_id = self.cover_canvas.create_text(
            20, self.center_xy, text="0:00", fill=self.ACCENT_CYAN, font=("Segoe UI", 9, "bold"), tags="layer7"
        )
        self.time_total_id = self.cover_canvas.create_text(
            self.canvas_size - 20, self.center_xy, text="-0:00", fill=self.FG_SECONDARY, font=("Segoe UI", 9), tags="layer7"
        )

        # 3. Text Info (Artist, Title, Album)
        info_frame = tk.Frame(self.wrapper, bg=self.BG_COLOR)
        info_frame.pack(fill="x", pady=(5, 10))
        
        self.song_artist_var = tk.StringVar(value="-")
        self.song_title_var = tk.StringVar(value="-")
        self.song_album_var = tk.StringVar(value="-")

        tk.Label(info_frame, textvariable=self.song_artist_var, font=("Segoe UI", 10), bg=self.BG_COLOR, fg=self.FG_SECONDARY).pack()
        tk.Label(info_frame, textvariable=self.song_title_var, font=("Segoe UI", 16, "bold"), bg=self.BG_COLOR, fg=self.ACCENT_CYAN).pack(pady=(2, 2))
        tk.Label(info_frame, textvariable=self.song_album_var, font=("Segoe UI", 9), bg=self.BG_COLOR, fg=self.FG_SECONDARY).pack()

        # 4. Main Transport Controls
        controls = tk.Frame(self.wrapper, bg=self.BG_COLOR)
        controls.pack(pady=(0, 20))

        btn_args = {"bg": self.BG_COLOR, "fg": self.FG_PRIMARY, "bd": 0, "activebackground": self.BG_COLOR, "activeforeground": self.ACCENT_CYAN, "cursor": "hand2"}

        tk.Button(controls, text="|◁", command=self._on_previous, font=("Segoe UI", 18), **btn_args).grid(row=0, column=0, padx=30)
        
        self.play_stop_button = tk.Button(
            controls, text="||", command=self._on_play_stop, 
            bg=self.BG_COLOR, fg=self.FG_PRIMARY, bd=0, 
            activebackground=self.BG_COLOR, activeforeground=self.ACCENT_CYAN, 
            font=("Segoe UI", 24, "bold"), cursor="hand2"
        )
        self.play_stop_button.grid(row=0, column=1, padx=30)
        
        tk.Button(controls, text="▷|", command=self._on_next, font=("Segoe UI", 18), **btn_args).grid(row=0, column=2, padx=30)

        # 5. Bottom Section Container (Log and Playlist)
        bottom_frame = tk.Frame(self.wrapper, bg=self.BG_DARK, padx=20, pady=15)
        bottom_frame.pack(fill="both", expand=True)
        
        # Event Log (Packed at the bottom so it never gets pushed off screen)
        log_frame = tk.Frame(bottom_frame, bg=self.BG_DARK)
        log_frame.pack(side="bottom", fill="x", pady=(10, 0))
        
        tk.Label(log_frame, text="EVENT LOG", font=("Segoe UI", 8, "bold"), bg=self.BG_DARK, fg=self.FG_SECONDARY, anchor="w").pack(fill="x", pady=(0, 5))
        self.log = tk.Text(
            log_frame, bg=self.BG_DARK, fg=self.FG_SECONDARY, bd=0, highlightthickness=0, 
            wrap="word", font=("Segoe UI", 8), height=3
        )
        self.log.pack(fill="x")

        # Scrollable Playlist Area (Takes up remaining vertical space above the log)
        playlist_wrapper = tk.Frame(bottom_frame, bg=self.BG_DARK)
        playlist_wrapper.pack(side="top", fill="both", expand=True)

        self.playlist_canvas = tk.Canvas(playlist_wrapper, bg=self.BG_DARK, highlightthickness=0)
        
        # Use our newly themed ttk.Scrollbar
        self.scrollbar = ttk.Scrollbar(playlist_wrapper, orient="vertical", command=self.playlist_canvas.yview, style="Custom.Vertical.TScrollbar")
        self.playlist_container = tk.Frame(self.playlist_canvas, bg=self.BG_DARK)

        self.playlist_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side="right", fill="y")
        self.playlist_canvas.pack(side="left", fill="both", expand=True)
        
        # Create a window inside the canvas for the frame
        self.canvas_window_id = self.playlist_canvas.create_window((0, 0), window=self.playlist_container, anchor="nw")
        
        # Update scrollregion automatically when the playlist container changes size
        self.playlist_container.bind("<Configure>", lambda e: self.playlist_canvas.configure(scrollregion=self.playlist_canvas.bbox("all")))
        # Keep the playlist items stretching to the canvas width
        self.playlist_canvas.bind("<Configure>", lambda e: self.playlist_canvas.itemconfig(self.canvas_window_id, width=e.width))

        # Mousewheel binding for smooth scrolling
        def _on_mousewheel(event):
            # Check delta for Windows, fall back to -1/1 for other platforms if needed
            delta = int(-1*(event.delta/120)) if event.delta else 0
            if delta == 0:
                delta = -1 if event.delta > 0 else 1
            self.playlist_canvas.yview_scroll(delta, "units")

        self.playlist_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._append_log("UI ready. Speak commands directly: play, stop, next, previous.")
        if not HAS_PIL:
            self._append_log("NOTICE: Install Pillow (`pip install pillow`) for images.")

        # Floating Toast for Voice Commands
        self.toast_label = tk.Label(
            self.root, text="", bg=self.ACCENT_CYAN, fg="#000000", 
            font=("Segoe UI", 11, "bold"), padx=15, pady=8, bd=0
        )
        self.toast_label.place_forget()
        
        # Update the album art immediately so it shows up on app launch
        self._update_album_art()
        
    def _render_playlist(self) -> None:
        """Dynamically renders the UP NEXT list inside the scrollable canvas."""
        # Clear existing items
        for widget in self.playlist_container.winfo_children():
            widget.destroy()

        playlist = self.player.playlist
        if len(playlist) <= 1:
            return

        current_idx = self.player.current_index

        def create_header(text: str, top_pad: int):
            tk.Label(self.playlist_container, text=text, font=("Segoe UI", 8, "bold"), bg=self.BG_DARK, fg=self.FG_SECONDARY, anchor="w").pack(fill="x", pady=(top_pad, 2))
            tk.Frame(self.playlist_container, bg="#2A2E38", height=1).pack(fill="x", pady=(0, 8))

        def create_item(index: int):
            track = playlist[index]
            item_frame = tk.Frame(self.playlist_container, bg=self.BG_DARK)
            item_frame.pack(fill="x", pady=2)

            # Left thumbnail
            if self.thumb_tk_image:
                lbl_thumb = tk.Label(item_frame, image=self.thumb_tk_image, bg=self.BG_DARK, bd=0)
                lbl_thumb.pack(side="left", padx=(0, 12))

            # Middle text
            text_frame = tk.Frame(item_frame, bg=self.BG_DARK)
            text_frame.pack(side="left", fill="x", expand=True)

            lbl_title = tk.Label(text_frame, text=track.title, font=("Segoe UI", 10, "bold"), fg=self.ACCENT_CYAN, bg=self.BG_DARK, anchor="w")
            lbl_title.pack(fill="x")

            dur = self._seconds_to_mmss(track.duration_seconds)
            lbl_sub = tk.Label(text_frame, text=f"{track.artist} • {dur}", font=("Segoe UI", 8), fg=self.FG_SECONDARY, bg=self.BG_DARK, anchor="w")
            lbl_sub.pack(fill="x")

            # Right play button
            btn_play = tk.Button(
                item_frame, text="▷", font=("Segoe UI", 16), fg=self.FG_PRIMARY, bg=self.BG_DARK, bd=0, 
                activebackground=self.BG_DARK, activeforeground=self.ACCENT_CYAN, cursor="hand2", 
                command=lambda i=index: self._play_from_list(i)
            )
            btn_play.pack(side="right", padx=(10, 0))

        # Render elements
        create_header("UP NEXT", 0)
        
        # Loop through all remaining songs in order and display them
        for i in range(1, len(playlist)):
            next_idx = (current_idx + i) % len(playlist)
            create_item(next_idx)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _show_command_toast(self, transcript: str) -> None:
        self.toast_label.configure(text=f"🗣️ {format_transcript_for_display(transcript)}")
        self.toast_label.place(relx=0.5, rely=0.15, anchor="center")

        if self.toast_hide_job is not None:
            self.root.after_cancel(self.toast_hide_job)
        self.toast_hide_job = self.root.after(2200, self._hide_command_toast)

    def _hide_command_toast(self) -> None:
        self.toast_label.place_forget()
        self.toast_hide_job = None

    def _seconds_to_mmss(self, seconds: int) -> str:
        minutes = seconds // 60
        remaining = seconds % 60
        return f"{minutes}:{remaining:02d}"
        
    def _spin_tick(self) -> None:
        """Runs fast to update image rotation smoothly."""
        if self.player.state == "playing":
            # Just spin it continuously at a set speed
            self.rotation_angle = (self.rotation_angle - 1.5) % 360
            self._update_album_art()
        self.root.after(50, self._spin_tick)

    def _update_album_art(self) -> None:
        if HAS_PIL and self.original_image:
            rotated = self.original_image.rotate(self.rotation_angle, resample=Image.Resampling.BICUBIC)
            self.tk_image = ImageTk.PhotoImage(rotated)
            self.cover_canvas.itemconfig(self.album_art_id, image=self.tk_image)

    def _refresh_ui(self) -> None:
        track = self.player.current_track
        
        self.song_artist_var.set(track.artist)
        self.song_title_var.set(track.title)
        self.song_album_var.set(track.album)

        total = track.duration_seconds
        current = min(self.player.position_seconds, total)
        remaining = total - current
        
        self.cover_canvas.itemconfig(self.time_current_id, text=f"{self._seconds_to_mmss(current)}")
        self.cover_canvas.itemconfig(self.time_total_id, text=f"-{self._seconds_to_mmss(remaining)}")
        
        # Calculate circular progress
        percentage = current / total if total > 0 else 0
        extent_degrees = -(percentage * 360) 

        # Draw the solid matching cyan arc
        self.cover_canvas.delete("progress_arcs")
        if extent_degrees != 0:
            self.cover_canvas.create_arc(
                self.center_xy - self.progress_radius, self.center_xy - self.progress_radius,
                self.center_xy + self.progress_radius, self.center_xy + self.progress_radius,
                start=180, extent=extent_degrees, 
                outline=self.ACCENT_CYAN, width=4, style="arc", tags="progress_arcs"
            )
        
        # Calculate and place the little progress dot at the end of the arc
        angle_rad = math.pi - (percentage * 2 * math.pi) # math.pi is 180 deg
        dot_x = self.center_xy + self.progress_radius * math.cos(angle_rad)
        dot_y = self.center_xy - self.progress_radius * math.sin(angle_rad)
        
        dot_radius = 5
        self.cover_canvas.coords(
            self.progress_dot, 
            dot_x - dot_radius, dot_y - dot_radius, 
            dot_x + dot_radius, dot_y + dot_radius
        )

        # Ensure correct layer drawing order
        self.cover_canvas.tag_lower("layer1") # Album Art
        self.cover_canvas.tag_raise("layer3") # Center Vinyl Hole
        self.cover_canvas.tag_raise("layer4") # Track Background
        self.cover_canvas.tag_raise("progress_arcs") # Solid Progress
        self.cover_canvas.tag_raise("layer6") # Progress Dot
        self.cover_canvas.tag_raise("layer7") # Time Text

        self.play_stop_button.configure(text="||" if self.player.state == "playing" else "▷")

        # Update Playlist UI efficiently (only rebuild if track changes)
        if self.last_rendered_index != self.player.current_index:
            self._render_playlist()
            self.last_rendered_index = self.player.current_index

    def _timer_tick(self) -> None:
        changed = self.player.tick()
        if changed:
            self._append_log(f"Auto-next: {self.player.current_track.title}")
            self.rotation_angle = 0.0

        self._refresh_ui()
        self.root.after(1000, self._timer_tick)

    def _on_close(self) -> None:
        self.voice_running = False
        self.audio_backend.stop()
        self.root.destroy()

    def _on_previous(self) -> None:
        self.rotation_angle = 0.0
        self._append_log(self.player.previous())
        self._refresh_ui()

    def _on_play_stop(self) -> None:
        if self.player.state == "playing":
            message = self.player.pause()
        else:
            # this will auto-resume if paused thanks to our SimulatedPlayer logic update
            message = self.player.play()
            
        self._append_log(message)
        self._refresh_ui()

    def _on_next(self) -> None:
        self.rotation_angle = 0.0
        self._append_log(self.player.next())
        self._refresh_ui()

    def _play_from_list(self, index: int) -> None:
        self.player.current_index = index
        self.player.position_seconds = 0
        self.player.state = "stopped" # Force fresh play
        self.rotation_angle = 0.0
        message = self.player.play()
        self._append_log(f"Selected: {self.player.current_track.title}")
        self._append_log(message)
        self._refresh_ui()

    def _start_hands_free_voice_loop(self) -> None:
        if sr is None:
            return

        if self.voice_running:
            return

        self.voice_running = True
        self._append_log("Hands-free voice enabled.")
        worker = threading.Thread(target=self._continuous_voice_worker, daemon=True)
        worker.start()

    def _continuous_voice_worker(self) -> None:
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
        try:
            while True:
                kind, value = self.queue.get_nowait()

                if kind == "transcript":
                    transcript = value.strip()
                    if not transcript:
                        continue

                    now = time.time()
                    duplicate = transcript.lower() == self.last_voice_transcript.lower()
                    if duplicate and (now - self.last_voice_time) < self.voice_cooldown_seconds:
                        continue

                    self.last_voice_transcript = transcript
                    self.last_voice_time = now

                    display_transcript = format_transcript_for_display(transcript)
                    self._append_log(f"Voice Command: {display_transcript}")
                    self._show_command_toast(display_transcript)
                    self._handle_command(transcript)
                else:
                    self._append_log(f"Voice error: {value}")

        except queue.Empty:
            pass

        self.root.after(150, self._poll_worker_queue)

    def _handle_command(self, transcript: str) -> None:
        lowered = transcript.lower().strip()
        display_transcript = format_transcript_for_display(transcript)

        if re.search(r"\b(previous|prev|back)\b", lowered):
            self.rotation_angle = 0.0
            self._append_log(f"Command: {display_transcript}")
            self._append_log(self.player.previous())
            self._refresh_ui()
            return

        interpretation = interpret_command(
            transcript,
            [{"title": t.title, "artist": t.artist, "album": t.album} for t in self.player.playlist if t.file_path],
        )

        self._append_log(f"Command: {display_transcript}")

        if interpretation.intent == "play":
                    # If we asked for a specific song, set it and reset the rotation
                    if interpretation.matched_song:
                        self.player.set_track_by_title(interpretation.matched_song.title)
                        self.rotation_angle = 0.0
                    # If we are just saying "play" to start from a stopped state, reset it
                    elif self.player.state != "paused":
                        self.rotation_angle = 0.0
                    # (If it IS paused, we do nothing to the angle so it resumes smoothly)

                    message = self.player.play()
                    if interpretation.song_query and not interpretation.matched_song:
                        message += f" | No match for '{interpretation.song_query}'"
                    self._append_log(message)
        elif interpretation.intent == "pause":
            self._append_log(self.player.pause())
        elif interpretation.intent == "stop":
            self._append_log(self.player.stop())
        elif interpretation.intent == "resume":
            self._append_log(self.player.play())
        elif interpretation.intent == "next":
            self.rotation_angle = 0.0
            self._append_log(self.player.next())
        else:
            self._append_log("Unknown command. Try: play, stop, next.")

        self._refresh_ui()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tkinter music simulation UI")
    parser.add_argument("--music-dir", type=Path, default=None, help="Optional music folder override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    PlayerUI(root, music_dir=args.music_dir)
    root.mainloop()


if __name__ == "__main__":
    main()