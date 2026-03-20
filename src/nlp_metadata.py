"""Lightweight NLP metadata search for voice-command music control.

This module provides a small text-processing layer that:
1) infers command intent from transcribed text,
2) extracts optional song title hints,
3) searches a local song metadata catalog with simple fuzzy ranking.

The implementation intentionally avoids heavy NLP dependencies so it can run
quickly in classroom and demo environments.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# Intent aliases used to map free-text commands to canonical playback intents.
INTENT_ALIASES: dict[str, tuple[str, ...]] = {
    "play": ("play", "resume", "start"),
    "pause": ("pause", "hold", "wait"),
    "next": ("next", "skip"),
    "stop": ("stop", "end", "quit"),
}


# A tiny fallback catalog for local testing when no catalog JSON is provided.
DEFAULT_CATALOG = [
    {"title": "Shape of You", "artist": "Ed Sheeran", "album": "Divide"},
    {"title": "Blinding Lights", "artist": "The Weeknd", "album": "After Hours"},
    {"title": "Levitating", "artist": "Dua Lipa", "album": "Future Nostalgia"},
    {"title": "As It Was", "artist": "Harry Styles", "album": "Harry's House"},
]


@dataclass(slots=True)
class SearchResult:
    """Represents one ranked metadata search hit."""

    title: str
    artist: str
    album: str
    score: float


@dataclass(slots=True)
class CommandInterpretation:
    """Structured interpretation of a free-form voice transcript."""

    transcript: str
    intent: str
    song_query: str | None
    matched_song: SearchResult | None


def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation for robust keyword matching."""

    lowered = text.lower().strip()
    return re.sub(r"[^a-z0-9\s]", " ", lowered)


def collapse_spaces(text: str) -> str:
    """Replace repeated spaces with a single space."""

    return re.sub(r"\s+", " ", text).strip()


def detect_intent(transcript: str) -> str:
    """Map transcript to one of the canonical intents or 'unknown'."""

    normalized = normalize_text(transcript)
    tokens = set(collapse_spaces(normalized).split())

    # Any alias hit maps directly to the canonical intent.
    for intent, aliases in INTENT_ALIASES.items():
        if any(alias in tokens for alias in aliases):
            return intent
    return "unknown"


def extract_song_query(transcript: str, intent: str) -> str | None:
    """Extract probable song title phrase from play-like commands."""

    if intent != "play":
        return None

    normalized = collapse_spaces(normalize_text(transcript))

    # Remove leading intent words and common connector words so the remaining
    # text is mostly the requested song phrase.
    normalized = re.sub(r"^(play|resume|start)\s+", "", normalized)
    normalized = re.sub(r"^(song|music|track)\s+", "", normalized)
    normalized = re.sub(r"^(called|named|title)\s+", "", normalized)

    # Handle phrases such as "play shape of you by ed sheeran".
    normalized = re.sub(r"\s+by\s+.+$", "", normalized)
    normalized = collapse_spaces(normalized)

    return normalized if normalized else None


def token_overlap_score(query: str, candidate_title: str, candidate_artist: str) -> float:
    """Compute a token-overlap score between query and metadata fields."""

    query_tokens = set(collapse_spaces(normalize_text(query)).split())
    title_tokens = set(collapse_spaces(normalize_text(candidate_title)).split())
    artist_tokens = set(collapse_spaces(normalize_text(candidate_artist)).split())

    if not query_tokens:
        return 0.0

    title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
    artist_overlap = len(query_tokens & artist_tokens) / len(query_tokens)
    return 0.8 * title_overlap + 0.2 * artist_overlap


def fuzzy_ratio(query: str, candidate_text: str) -> float:
    """Compute sequence similarity score in the range [0, 1]."""

    return SequenceMatcher(None, normalize_text(query), normalize_text(candidate_text)).ratio()


def rank_catalog(query: str, catalog: list[dict[str, str]], top_k: int = 3) -> list[SearchResult]:
    """Rank catalog entries using token overlap + fuzzy matching."""

    scored: list[SearchResult] = []
    for item in catalog:
        title = item.get("title", "")
        artist = item.get("artist", "")
        album = item.get("album", "")

        overlap = token_overlap_score(query, title, artist)
        title_fuzzy = fuzzy_ratio(query, title)
        joint_fuzzy = fuzzy_ratio(query, f"{title} {artist}")

        # Weighted blend favors title match while still considering artist text.
        final_score = 0.5 * overlap + 0.35 * title_fuzzy + 0.15 * joint_fuzzy
        scored.append(
            SearchResult(
                title=title,
                artist=artist,
                album=album,
                score=round(final_score, 6),
            )
        )

    # Highest score first so caller can take top-1 or top-k confidently.
    scored.sort(key=lambda row: row.score, reverse=True)
    return scored[:top_k]


def load_catalog(path: Path | None) -> list[dict[str, str]]:
    """Load metadata catalog from JSON, or fallback to built-in examples."""

    if path is None:
        return DEFAULT_CATALOG.copy()

    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Catalog JSON must be a list of objects.")

    return [
        {
            "title": str(item.get("title", "")),
            "artist": str(item.get("artist", "")),
            "album": str(item.get("album", "")),
        }
        for item in payload
    ]


def interpret_command(
    transcript: str,
    catalog: list[dict[str, str]],
    min_score: float = 0.45,
) -> CommandInterpretation:
    """Interpret transcript and optionally resolve best song metadata match."""

    # Pipeline: intent detection -> optional song query extraction -> catalog
    # ranking -> confidence gate.
    intent = detect_intent(transcript)
    song_query = extract_song_query(transcript, intent)

    matched_song: SearchResult | None = None
    if intent == "play" and song_query:
        ranked = rank_catalog(song_query, catalog, top_k=1)
        if ranked and ranked[0].score >= min_score:
            matched_song = ranked[0]

    return CommandInterpretation(
        transcript=transcript,
        intent=intent,
        song_query=song_query,
        matched_song=matched_song,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI options for quick local testing."""

    parser = argparse.ArgumentParser(description="Run lightweight NLP metadata search.")
    parser.add_argument("--text", type=str, required=True, help="Transcribed command text.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Optional JSON catalog file with title/artist/album records.",
    )
    parser.add_argument("--min-score", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    """CLI entry point for metadata-search demonstration."""

    args = parse_args()
    catalog = load_catalog(args.catalog)
    interpretation = interpret_command(
        transcript=args.text,
        catalog=catalog,
        min_score=args.min_score,
    )

    payload: dict[str, Any] = asdict(interpretation)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
