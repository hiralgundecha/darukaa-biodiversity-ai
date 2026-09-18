"""Multi-turn state.

A session accumulates a SiteProfile across turns. Values arrive either as
structured JSON or as free text, and both paths merge into the same profile,
so a user can start by typing a sentence and later post a JSON block without
losing what was already established.

Storage is in-process by design for the demo. Swapping the dict for a Postgres
table is a contained change: only get() and save() touch it.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from .schemas import SiteProfile

LAND_USE_WORDS = {
    "monoculture": "monoculture",
    "single crop": "monoculture",
    "wheat": "monoculture",
    "paddy": "monoculture",
    "rice": "monoculture",
    "sugarcane": "monoculture",
    "cropland": "cropland",
    "farm": "cropland",
    "field": "cropland",
    "agriculture": "cropland",
    "grassland": "grassland",
    "pasture": "pasture",
    "grazing": "pasture",
    "plantation": "plantation",
    "coffee": "plantation",
    "tea": "plantation",
    "degraded": "degraded",
    "wasteland": "degraded",
    "barren": "degraded",
    "forest": "forest",
    "mangrove": "coastal",
    "creek": "coastal",
    "coastal": "coastal",
    "wetland": "wetland",
    "agroforestry": "agroforestry",
}

CLIMATE_WORDS = ["semi-arid", "semi arid", "sub-humid", "sub humid", "arid", "humid", "tropical", "coastal"]
LEVELS = ["low", "moderate", "high"]


def _level_near(text: str, *keywords: str) -> str | None:
    """Find a low/moderate/high word within a short window of a keyword.

    Nearest match wins rather than first match, because a sentence often carries
    two levels ("rainfall is low but pollution is high") and taking the first
    level word in the window assigns both keywords the same value.
    """
    best: tuple[int, str] | None = None
    for keyword in keywords:
        for match in re.finditer(re.escape(keyword), text):
            centre = (match.start() + match.end()) // 2
            for level in LEVELS:
                for lm in re.finditer(rf"\b{level}\b", text):
                    distance = abs(((lm.start() + lm.end()) // 2) - centre)
                    if distance <= 45 and (best is None or distance < best[0]):
                        best = (distance, level)
            for pattern, level in (
                (r"\b(scarce|poor|deficient|little|declin)\w*", "low"),
                (r"\b(heavy|severe|intense|excess)\w*", "high"),
            ):
                for lm in re.finditer(pattern, text):
                    distance = abs(((lm.start() + lm.end()) // 2) - centre)
                    if distance <= 45 and (best is None or distance < best[0]):
                        best = (distance, level)
    if best:
        return best[1]
    return None


def extract_from_text(text: str) -> dict[str, Any]:
    """Pull whatever structured values a sentence happens to contain.

    Regex rather than an LLM call on purpose: extraction failures here are
    silent and cheap to audit, and every extracted value is echoed back to the
    user in the reply so a wrong parse is visible immediately.
    """
    t = text.lower()
    out: dict[str, Any] = {}

    m = re.search(r"(?:organic carbon|soc|carbon)[^\d%]{0,25}(\d+(?:\.\d+)?)\s*%", t)
    if m:
        out["soil_organic_carbon_pct"] = float(m.group(1))

    m = re.search(r"\bph\b[^\d]{0,15}(\d+(?:\.\d+)?)", t)
    if m:
        out["soil_ph"] = float(m.group(1))

    m = re.search(r"(\d{2,5})\s*mm", t)
    if m:
        out["rainfall_mm"] = float(m.group(1))

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:ha|hectare)", t)
    if m:
        out["area_ha"] = float(m.group(1))

    m = re.search(r"(-?\d{1,2}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)", t)
    if m:
        out["lat"], out["lon"] = float(m.group(1)), float(m.group(2))

    m = re.search(r"(\d+)\s*(?:species|taxa)", t)
    if m:
        out["species_richness"] = int(m.group(1))

    for word, value in LAND_USE_WORDS.items():
        if word in t:
            out["land_use"] = value
            break

    for word in CLIMATE_WORDS:
        if word in t:
            out["climate"] = word.replace(" ", "-").replace("tropical", "humid")
            break

    rainfall = _level_near(t, "rainfall", "rain", "monsoon", "precipitation")
    if rainfall:
        out["rainfall"] = rainfall
    moisture = _level_near(t, "moisture", "soil water", "dry")
    if moisture:
        out["soil_moisture"] = moisture
    pollution = _level_near(t, "pollution", "pesticide", "fertiliser", "fertilizer", "effluent", "runoff")
    if pollution:
        out["pollution"] = pollution
    frag = _level_near(t, "fragment", "isolated", "patches", "connectivity")
    if frag:
        out["habitat_fragmentation"] = frag
    defo = _level_near(t, "deforest", "tree loss", "felling", "clearing")
    if defo:
        out["deforestation"] = defo

    if "steep" in t:
        out["slope"] = "steep"
    elif "sloping" in t or "slope" in t:
        out["slope"] = "moderate"
    elif "flat" in t:
        out["slope"] = "flat"

    return out


@dataclass
class Session:
    session_id: str
    site: SiteProfile = field(default_factory=SiteProfile)
    turns: list[dict[str, str]] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)

    def merge(self, values: dict[str, Any]) -> None:
        """Later turns overwrite earlier ones. A user correcting themselves
        should not have to start a new session."""
        current = self.site.model_dump()
        current.update({k: v for k, v in values.items() if v is not None})
        self.site = SiteProfile(**current)


_SESSIONS: dict[str, Session] = {}


def get(session_id: str | None) -> Session:
    if session_id and session_id in _SESSIONS:
        return _SESSIONS[session_id]
    new_id = session_id or uuid.uuid4().hex[:12]
    session = Session(session_id=new_id)
    _SESSIONS[new_id] = session
    return session


def save(session: Session) -> None:
    _SESSIONS[session.session_id] = session


def reset(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)
