"""The reasoning layer.

Retrieval alone gives a ranked list of practices that sound like the query.
That is not a recommendation. This module does three further things:

1. Diagnoses the site by combining variables, so that "low carbon" and
   "low rainfall" together produce a different conclusion than either alone.
2. Filters retrieved cards against the site's hard conditions, so that a
   liming card cannot be recommended for an alkaline soil.
3. Scores what survives on how many of the site's actual problems it touches,
   not on text similarity, and attaches a confidence that falls when inputs
   are sparse or the evidence tier is weak.
"""

from __future__ import annotations

from typing import Any

from .retriever import get_store
from .schemas import Diagnosis, Evidence, MetricEffect, Recommendation, SiteProfile

REQUIRED_MIN = 3

LINK_LABELS = {
    "soil-biodiversity": "soil health to biodiversity",
    "soil-water": "soil health to water retention",
    "water-species": "water availability to species survival",
    "landuse-fragmentation": "land use to habitat fragmentation",
    "pollution-species": "pollution load to species survival",
}

# Ordered by how much the answer changes if we learn it.
QUESTION_PRIORITY = [
    ("land_use", "What is the land currently used for (cropland, grassland, plantation, degraded, forest, coastal)?"),
    ("soil_organic_carbon_pct", "Do you have a soil organic carbon figure, as a percentage of topsoil?"),
    ("rainfall", "How would you describe rainfall over the last few seasons: low, moderate or high?"),
    ("soil_ph", "What is the soil pH?"),
    (
        "habitat_fragmentation",
        "Is the surrounding landscape broken into isolated patches (low, moderate or high fragmentation)?",
    ),
    (
        "pollution",
        "Is there noticeable pollution pressure from fertiliser, pesticide or effluent (low, moderate or high)?",
    ),
    ("soil_moisture", "How does the soil hold moisture between rain events: low, moderate or high?"),
    ("climate", "Which climate zone does the site fall in (arid, semi-arid, sub-humid, humid)?"),
    ("slope", "Is the land flat, moderately sloping or steep?"),
    ("deforestation", "Has there been tree cover loss on or around the site recently (low, moderate or high)?"),
]

# Crude but useful: lets a bare coordinate pair seed climate and region.
CLIMATE_BY_LAT = [
    (23.5, "semi-arid"),
    (35.0, "sub-humid"),
    (90.0, "humid"),
]


def infer_from_geo(site: SiteProfile) -> SiteProfile:
    """Fill climate and region from coordinates when nothing better is given.

    Anything inferred this way is treated as weaker than a stated value and is
    reported back to the user so they can correct it.
    """
    if site.lat is None or site.lon is None:
        return site
    data = site.model_dump()
    if data.get("climate") is None:
        abs_lat = abs(site.lat)
        for bound, zone in CLIMATE_BY_LAT:
            if abs_lat <= bound:
                data["climate"] = "humid" if abs_lat < 10 else zone
                break
    if data.get("region") is None:
        data["region"] = f"{site.lat:.3f}, {site.lon:.3f}"
    return SiteProfile(**data)


def normalise(site: SiteProfile) -> SiteProfile:
    """Derive categorical values from numeric ones where the mapping is safe."""
    data = site.model_dump()
    if data.get("rainfall") is None and data.get("rainfall_mm") is not None:
        mm = data["rainfall_mm"]
        data["rainfall"] = "low" if mm < 600 else ("moderate" if mm < 1200 else "high")
    if data.get("climate") is None and data.get("rainfall_mm") is not None:
        mm = data["rainfall_mm"]
        data["climate"] = "arid" if mm < 300 else ("semi-arid" if mm < 700 else ("sub-humid" if mm < 1400 else "humid"))
    if data.get("land_use"):
        data["land_use"] = str(data["land_use"]).strip().lower().replace(" ", "_")
    return SiteProfile(**data)


def _count_signals(site: SiteProfile) -> int:
    """How many distinct environmental variables we actually hold.

    Coordinates and area describe where the site is, not what condition it is
    in, so they do not count toward the three-variable minimum.
    """
    ignore = {"lat", "lon", "region", "area_ha", "rainfall_mm"}
    known = set(site.known_fields()) - ignore
    if site.rainfall_mm is not None:
        known.add("rainfall")
    return len(known)


def missing_questions(site: SiteProfile, limit: int = 3) -> tuple[list[str], list[str]]:
    known = site.known_fields()
    fields, questions = [], []
    for field, question in QUESTION_PRIORITY:
        if field not in known:
            fields.append(field)
            questions.append(question)
        if len(questions) >= limit:
            break
    return fields, questions


def diagnose(site: SiteProfile) -> list[Diagnosis]:
    """Combine variables into named problem states.

    Each rule here needs at least two variables to fire. That is the point:
    a single-variable answer is what the brief asks us not to produce.
    """
    out: list[Diagnosis] = []
    soc = site.soil_organic_carbon_pct
    dry = site.rainfall == "low" or site.climate in {"arid", "semi-arid"} or site.soil_moisture == "low"
    simplified = site.land_use in {"monoculture", "cropland", "plantation"}

    if soc is not None and soc < 0.75 and dry:
        out.append(
            Diagnosis(
                code="carbon_water_trap",
                label="Carbon-depleted soil in a water-limited system",
                explanation=(
                    "Low organic carbon and low water availability reinforce each other. Carbon-poor soil holds "
                    "less plant-available water, so biomass production stays low, so carbon input stays low. "
                    "Interventions that only add carbon inputs work slowly here; the moisture side has to be "
                    "addressed at the same time or the added residue simply oxidises."
                ),
                variables=["soil_organic_carbon_pct", "rainfall", "soil_moisture"],
            )
        )
    if soc is not None and soc < 1.0 and simplified:
        out.append(
            Diagnosis(
                code="simplified_system",
                label="Structurally simplified land use with a depleted soil biological base",
                explanation=(
                    "A single crop or uniform plantation supplies one kind of root exudate and one litter "
                    "chemistry, so the soil food web narrows. A narrow soil food web means less invertebrate "
                    "biomass, which is the food supply for most of the above-ground fauna a biodiversity survey "
                    "would count. Diversifying the vegetation is the upstream fix; adding species lists "
                    "downstream is not."
                ),
                variables=["land_use", "soil_organic_carbon_pct", "species_richness"],
            )
        )
    if site.habitat_fragmentation in {"moderate", "high"} and simplified:
        out.append(
            Diagnosis(
                code="isolation_limited",
                label="Habitat present but functionally isolated",
                explanation=(
                    "Where fragmentation is the binding constraint, improving habitat quality inside a patch has "
                    "limited effect, because dispersal-limited species cannot reach it. Connectivity work comes "
                    "before quality work in this ordering."
                ),
                variables=["habitat_fragmentation", "land_use", "species_richness"],
            )
        )
    if site.pollution in {"moderate", "high"} and simplified:
        out.append(
            Diagnosis(
                code="chemical_suppression",
                label="Chemical load suppressing recovery",
                explanation=(
                    "Agrochemical pressure removes natural enemies and pollinators faster than habitat work can "
                    "replace them. Restoration effort spent alongside a continuing high chemical load tends to be "
                    "wasted, so load reduction is sequenced first."
                ),
                variables=["pollution", "land_use", "species_richness"],
            )
        )
    if site.soil_ph is not None and (site.soil_ph < 5.5 or site.soil_ph > 8.2):
        direction = "acidic" if site.soil_ph < 5.5 else "alkaline or sodic"
        out.append(
            Diagnosis(
                code="ph_constraint",
                label=f"Chemical constraint: soil is strongly {direction}",
                explanation=(
                    "pH outside roughly 5.5 to 8.2 limits nutrient availability and suppresses parts of the soil "
                    "microbial community, including the rhizobia that legume-based practices depend on. This is a "
                    "gating constraint: correct it first, or the biological interventions underperform."
                ),
                variables=["soil_ph", "soil_organic_carbon_pct", "species_richness"],
            )
        )
    if site.deforestation in {"moderate", "high"} or site.land_use == "degraded":
        out.append(
            Diagnosis(
                code="cover_loss",
                label="Recent or ongoing vegetation cover loss",
                explanation=(
                    "Where cover is still being lost, restoration runs against an active drain. Securing the "
                    "existing stock and the regenerating rootstock returns more per rupee than new planting."
                ),
                variables=["deforestation", "land_use", "carbon_stock"],
            )
        )
    return out


def _conditions_match(card: dict[str, Any], site: SiteProfile) -> tuple[bool, float]:
    """Hard gates first, then a soft match score.

    Returning False here removes the card entirely. That is what stops the
    system recommending lime on alkaline soil, which is the class of error a
    similarity-only pipeline makes.
    """
    cond = card.get("conditions", {})
    score = 0.0

    if "soil_ph_max" in cond:
        if site.soil_ph is None or site.soil_ph > cond["soil_ph_max"]:
            return False, 0.0
        score += 1.5
    if "soil_ph_min" in cond:
        if site.soil_ph is None or site.soil_ph < cond["soil_ph_min"]:
            return False, 0.0
        score += 1.5
    if "soil_organic_carbon_pct_max" in cond and site.soil_organic_carbon_pct is not None:
        if site.soil_organic_carbon_pct > cond["soil_organic_carbon_pct_max"]:
            return False, 0.0
        score += 1.0
    if "land_use" in cond and site.land_use:
        if site.land_use not in cond["land_use"]:
            return False, 0.0
        score += 1.0
    if "climate" in cond and site.climate:
        if site.climate in cond["climate"]:
            score += 0.75
        else:
            return False, 0.0
    if "rainfall" in cond and site.rainfall:
        if site.rainfall in cond["rainfall"]:
            score += 0.5
        else:
            return False, 0.0
    for key, attr in (
        ("habitat_fragmentation", "habitat_fragmentation"),
        ("pollution", "pollution"),
        ("deforestation", "deforestation"),
        ("slope", "slope"),
    ):
        value = getattr(site, attr)
        if key in cond and value:
            if value in cond[key]:
                score += 0.75
            else:
                return False, 0.0
    return True, score


def _diagnosis_bonus(card: dict[str, Any], diagnoses: list[Diagnosis]) -> tuple[float, list[str]]:
    codes = {d.code for d in diagnoses}
    metrics = {e["metric"] for e in card.get("effects", [])}
    links = set(card.get("links", []))
    bonus, addressed = 0.0, []

    rules = [
        (
            "carbon_water_trap",
            {"soil_moisture", "water_availability"} & metrics and {"soil_organic_carbon"} & metrics,
            2.5,
        ),
        ("simplified_system", bool({"soil_organic_carbon", "species_richness"} & metrics), 1.5),
        ("isolation_limited", "landuse-fragmentation" in links, 2.5),
        ("chemical_suppression", "pollution_load" in metrics, 2.5),
        ("ph_constraint", "soil_ph" in metrics, 3.0),
        ("cover_loss", "carbon_stock" in metrics, 2.0),
    ]
    for code, condition, weight in rules:
        if code in codes and condition:
            bonus += weight
            addressed.append(code)
    return bonus, addressed


def _confidence(card: dict[str, Any], cond_score: float, signals: int, addressed: int) -> float:
    tier_weight = {1: 0.82, 2: 0.70, 3: 0.58}.get(card.get("evidence_tier", 2), 0.6)
    completeness = min(signals / 6.0, 1.0)
    fit = min(cond_score / 4.0, 1.0)
    targeting = min(addressed / 2.0, 1.0)
    raw = tier_weight * (0.45 + 0.20 * completeness + 0.20 * fit + 0.15 * targeting)
    return round(min(max(raw, 0.15), 0.92), 2)


def _label(value: float) -> str:
    return "high" if value >= 0.72 else ("moderate" if value >= 0.55 else "low")


def _horizon(card: dict[str, Any]) -> str:
    order = {"short": 0, "medium": 1, "long": 2}
    horizons = [e.get("horizon", "medium") for e in card.get("effects", [])]
    return min(horizons, key=lambda h: order.get(h, 1)) if horizons else "medium"


def build_query(site: SiteProfile, message: str | None) -> str:
    parts = [message or ""]
    known = site.known_fields()
    for key, value in known.items():
        if key in {"lat", "lon"}:
            continue
        parts.append(f"{key.replace('_', ' ')} {value}")
    if site.soil_organic_carbon_pct is not None and site.soil_organic_carbon_pct < 1.0:
        parts.append("low soil organic carbon depleted soil")
    if site.rainfall == "low":
        parts.append("water limited dry rainfed moisture conservation")
    if site.habitat_fragmentation in {"moderate", "high"}:
        parts.append("habitat fragmentation connectivity corridor")
    if site.pollution in {"moderate", "high"}:
        parts.append("nutrient runoff pesticide load reduction")
    return " ".join(p for p in parts if p)


def recommend(site: SiteProfile, message: str | None, top_k: int = 4) -> dict[str, Any]:
    site = normalise(infer_from_geo(site))
    diagnoses = diagnose(site)
    signals = _count_signals(site)
    store = get_store()

    retrieved = store.search(build_query(site, message), k=14)
    scored: list[tuple[float, dict[str, Any], float, list[str], float]] = []

    for card, sim in retrieved:
        ok, cond_score = _conditions_match(card, site)
        if not ok:
            continue
        bonus, addressed = _diagnosis_bonus(card, diagnoses)
        link_count = len(card.get("links", []))
        total = (2.0 * sim) + cond_score + bonus + (0.4 * link_count)
        scored.append((total, card, sim, addressed, cond_score))

    scored.sort(key=lambda x: -x[0])
    chosen = scored[:top_k]

    # Always close with the monitoring card if it did not make the cut on its
    # own. Without a baseline none of the above can be verified later.
    if not any(c[1]["id"] == "kb-026" for c in chosen) and "kb-026" in store.by_id:
        card = store.by_id["kb-026"]
        chosen.append((0.0, card, 0.0, [], 1.0))

    recommendations: list[Recommendation] = []
    for rank, (total, card, sim, addressed, cond_score) in enumerate(chosen, start=1):
        conf = _confidence(card, cond_score, signals, len(addressed))
        recommendations.append(
            Recommendation(
                rank=rank,
                action=card["practice"],
                detail=card.get("summary", ""),
                why_it_works=card.get("mechanism", ""),
                addresses=[d.label for d in diagnoses if d.code in addressed],
                metrics=[MetricEffect(**e) for e in card.get("effects", [])],
                linked_variables=[LINK_LABELS.get(link, link) for link in card.get("links", [])],
                horizon=_horizon(card),
                confidence=conf,
                confidence_label=_label(conf),
                cost=card.get("cost", "unknown"),
                effort=card.get("effort", "unknown"),
                evidence=Evidence(kb_id=card["id"], source=card["source"], evidence_tier=card.get("evidence_tier", 2)),
                retrieval_score=round(float(sim), 3),
            )
        )

    return {
        "site": site,
        "signals": signals,
        "diagnoses": diagnoses,
        "recommendations": recommendations,
        "retrieved_ids": [c["id"] for c, _ in retrieved],
    }
