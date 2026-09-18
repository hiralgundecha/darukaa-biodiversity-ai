"""Tests cover the parts where a wrong answer is a scientific error, not a bug:
condition gating, multi-variable diagnosis, and the minimum-input rule."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.conversation import extract_from_text
from app.reasoning import _count_signals, diagnose, normalise, recommend
from app.retriever import get_store
from app.schemas import SiteProfile


def test_knowledge_base_loads():
    store = get_store()
    assert len(store.cards) >= 20
    for card in store.cards:
        assert card["source"]
        assert card["effects"]


def test_alkaline_soil_never_gets_lime():
    site = SiteProfile(soil_ph=8.9, land_use="cropland", rainfall="low", soil_organic_carbon_pct=0.4)
    ids = [r.evidence.kb_id for r in recommend(site, None)["recommendations"]]
    assert "kb-010" not in ids
    assert "kb-011" in ids


def test_acidic_soil_gets_lime():
    site = SiteProfile(soil_ph=4.9, land_use="cropland", rainfall="moderate", soil_organic_carbon_pct=0.6)
    ids = [r.evidence.kb_id for r in recommend(site, None)["recommendations"]]
    assert "kb-010" in ids


def test_carbon_water_trap_needs_both_variables():
    dry_only = SiteProfile(rainfall="low", land_use="cropland", climate="semi-arid")
    assert "carbon_water_trap" not in {d.code for d in diagnose(dry_only)}
    both = SiteProfile(rainfall="low", land_use="cropland", climate="semi-arid", soil_organic_carbon_pct=0.3)
    assert "carbon_water_trap" in {d.code for d in diagnose(both)}


def test_recommendations_are_multi_metric():
    site = SiteProfile(soil_organic_carbon_pct=0.3, rainfall="low", land_use="monoculture", climate="semi-arid")
    recs = recommend(site, "biodiversity is declining")["recommendations"]
    substantive = [r for r in recs if r.evidence.kb_id != "kb-026"]
    assert substantive
    assert all(len(r.metrics) >= 2 for r in substantive)
    assert any(r.linked_variables for r in substantive)


def test_every_recommendation_carries_a_source():
    site = SiteProfile(land_use="degraded", deforestation="high", rainfall="moderate", climate="sub-humid")
    for rec in recommend(site, None)["recommendations"]:
        assert rec.evidence.source
        assert rec.horizon in {"short", "medium", "long"}
        assert 0 < rec.confidence <= 1


def test_signal_count_ignores_location_only_fields():
    assert _count_signals(SiteProfile(lat=18.9, lon=73.1, area_ha=4)) == 0
    assert _count_signals(SiteProfile(lat=18.9, lon=73.1, rainfall_mm=500, land_use="cropland")) == 2


def test_text_extraction():
    got = extract_from_text("soil organic carbon is 0.3%, rainfall is low, monoculture wheat, pH 5.1")
    assert got["soil_organic_carbon_pct"] == 0.3
    assert got["rainfall"] == "low"
    assert got["land_use"] == "monoculture"
    assert got["soil_ph"] == 5.1


def test_rainfall_mm_derives_category():
    site = normalise(SiteProfile(rainfall_mm=420))
    assert site.rainfall == "low"
    assert site.climate == "semi-arid"


def test_windowed_levels_do_not_cross_contaminate():
    got = extract_from_text("rainfall is low but pesticide pollution is high")
    assert got["rainfall"] == "low"
    assert got["pollution"] == "high"
