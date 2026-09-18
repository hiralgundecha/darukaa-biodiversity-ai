# Darukaa Biodiversity Intelligence

A retrieval-grounded advisory system for land and biodiversity decisions. It holds a
sourced knowledge base of environmental interventions, works out what is actually
constraining a site by reading several variables together, and returns recommendations
that each carry a mechanism, the metrics they move, a time horizon, a confidence value
and a citation.

The design target in the brief was an environmental scientist rather than a chatbot, so
the language model is the smallest part of the system and is optional. Selection,
filtering and ranking are deterministic and testable; the model, when a key is present,
only rephrases decisions that have already been made.

---

## 1. How it works

```
user text or JSON
        │
        ▼
  extraction + session memory        conversation.py
        │   (merges turns into one SiteProfile)
        ▼
  enough variables?  ── no ──▶  ask the highest-value missing questions
        │ yes
        ▼
  diagnosis            reasoning.py   rules that each need ≥2 variables
        │
        ▼
  retrieval            retriever.py   cosine over embedded evidence cards
        │
        ▼
  hard condition gate  reasoning.py   removes cards the site contradicts
        │
        ▼
  scoring + confidence reasoning.py   similarity + fit + diagnosis coverage
        │
        ▼
  narration            narrate.py     template by default, LLM optional
```

### Why retrieval alone is not enough

Similarity search returns cards that read like the query. On a site with pH 8.9 it will
happily return the liming card, because liming and pH are textually close. That is a
scientific error, not a ranking imperfection, and no amount of reranking fixes it.

So retrieval is followed by a hard condition gate. Each card declares the conditions it
applies under (`soil_ph_max`, `land_use`, `climate`, and so on); a card whose conditions
the site contradicts is removed rather than demoted. There is a test for exactly this
case: `test_alkaline_soil_never_gets_lime`.

### Multi-variable diagnosis

Every diagnosis rule needs at least two variables before it fires. Low carbon on its own
produces generic soil advice. Low carbon plus low rainfall produces a different
conclusion, because the two reinforce each other: carbon-poor soil holds less water, so
biomass stays low, so carbon input stays low. The system names that state
(`carbon_water_trap`) and then prefers interventions that move both sides at once.

Current rules:

| Code | Fires on | Effect on ranking |
|---|---|---|
| `carbon_water_trap` | low SOC + dry conditions | favours cards affecting carbon and moisture together |
| `simplified_system` | monoculture or plantation + low SOC | favours vegetation diversification |
| `isolation_limited` | fragmentation + simplified land use | favours connectivity work ahead of patch quality |
| `chemical_suppression` | pollution pressure + simplified land use | favours load reduction first |
| `ph_constraint` | pH outside 5.5–8.2 | treated as gating; corrections rank above biological work |
| `cover_loss` | deforestation or degraded land use | favours protecting existing stock over new planting |

### Confidence

Confidence is computed, not asserted. It combines the evidence tier of the card, how
complete the site profile is, how well the card's conditions match, and how much of the
diagnosis the card addresses. It is capped at 0.92: a knowledge base of this size should
not be claiming certainty.

### Clarifying questions

Below three environmental variables the system refuses to recommend and asks instead.
Questions are ordered by how much the answer changes the ranking, not by schema order, so
land use and soil carbon are asked before slope. Coordinates and area do not count toward
the three, because they say where the site is rather than what condition it is in.

---

## 2. Knowledge base

26 evidence cards in `backend/data/knowledge.jsonl`, one JSON object per line:

```jsonc
{
  "id": "kb-001",
  "practice": "Legume-based cover cropping between main crop cycles",
  "summary": "...",
  "mechanism": "...",                  // the scientific reasoning shown to the user
  "conditions": { "land_use": [...], "soil_organic_carbon_pct_max": 1.2 },
  "effects": [ { "metric": "soil_organic_carbon", "direction": "increase",
                 "magnitude": "...", "horizon": "medium" } ],
  "links": ["soil-biodiversity", "soil-water"],
  "evidence_tier": 1,
  "source": "FAO, Conservation Agriculture and Soil Organic Carbon guidance; ..."
}
```

Coverage spans soil health, land use and land cover, biodiversity indicators, climate
factors, and human impact, which is what the brief asked the knowledge layer to hold.
Sources are FAO, IPCC AR6 WGIII Chapter 7, the IPBES global and thematic assessments,
and Indian watershed programme documentation.

Two honesty notes, because they affect how the output should be read. Magnitudes are
indicative ranges as reported in those assessments, not site-specific predictions, and
they are phrased as ranges for that reason. Citations name the report or programme rather
than a DOI, because the cards are compiled from assessment-level findings rather than from
individual papers. Indexing a paper corpus with per-passage citation is the obvious next
step and is listed under limitations.

---

## 3. Storage and retrieval

| Concern | Default | Optional upgrade |
|---|---|---|
| Vector store | in-memory numpy index | PostgreSQL 16 + pgvector, `ivfflat` cosine index |
| Embeddings | deterministic hashed bag-of-words, 384-dim | `all-MiniLM-L6-v2` via sentence-transformers |
| Narration | deterministic template | Anthropic API, given only the chosen recommendations |
| Session memory | in-process dict | contained behind `conversation.get/save` |

The default path was a deliberate choice. Everything works on a clone with four
dependencies and no Docker, no model download and no API key, and the optional backends
switch on from environment variables alone. The retrieval maths is the same either way,
so a reviewer running the light path sees the same rankings.

Hashing uses `zlib.crc32` rather than Python's `hash()`, because `hash()` on strings is
salted per process and the index would not be reproducible across restarts.

### pgvector schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE kb_cards (
    id         TEXT PRIMARY KEY,
    practice   TEXT NOT NULL,
    payload    JSONB NOT NULL,          -- the full card, so the API needs no second store
    embedding  vector(384) NOT NULL
);

CREATE INDEX kb_cards_embedding_idx
    ON kb_cards USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
```

The card body is kept in `JSONB` rather than normalised into tables. Conditions and
effects vary in shape between card types, and the query pattern is always "fetch the whole
card by id", so normalising would add joins without adding anything.

---

## 4. Running it locally

Needs Python 3.11 or newer. Nothing else is required.

```bash
git clone <this repo>
cd darukaa-biodiversity-ai

python -m venv .venv && source .venv/bin/activate
pip install fastapi "uvicorn[standard]" pydantic numpy

python backend/scripts/seed.py           # builds the index, prints the worked example
uvicorn app.main:app --app-dir backend --reload --port 8000
```

Open `http://localhost:8000` for the chat UI, or `http://localhost:8000/docs` for the
OpenAPI page.

Optional extras:

```bash
pip install -r backend/requirements.txt  # adds sentence-transformers, pgvector, anthropic
docker compose up -d db                  # pgvector on :5432
cp .env.example .env                     # set DATABASE_URL and/or ANTHROPIC_API_KEY
```

`GET /health` reports which backends actually came up, so a missing optional dependency
is visible rather than silent.

Tests:

```bash
pip install pytest ruff black
cd backend && python -m pytest tests -q
```

---

## 5. API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Conversational entry point. Accepts `message`, `site` (JSON), or both |
| `POST` | `/analyse` | Stateless structured input, for scripted or batch use |
| `GET` | `/search?q=` | Raw retrieval with no reasoning, for inspecting the index |
| `GET` | `/knowledge`, `/knowledge/{id}` | Browse the evidence cards |
| `GET` | `/health` | Card count and which backends are live |
| `DELETE` | `/session/{id}` | Clear session memory |

Text and JSON merge into the same profile, so a user can describe a site in a sentence and
then post a JSON block correcting one field without losing the rest.

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{
  "site": {"soil_organic_carbon_pct": 0.3, "rainfall": "low",
           "land_use": "monoculture", "climate": "semi-arid"},
  "message": "biodiversity is declining"
}'
```

Geo-coordinates are accepted and used to seed climate and region when nothing better is
given. Anything inferred that way is echoed back in `known` so the user can correct it.

---

## 6. Worked example

Input: soil organic carbon 0.3%, rainfall low, monoculture wheat, semi-arid.

Diagnosis: `carbon_water_trap` and `simplified_system`.

Top recommendation: legume-based cover cropping. Roughly 15–25% relative topsoil carbon
gain over two to three seasons, a within-season reduction in evaporative loss, and higher
soil invertebrate and pollinator counts. Short to medium horizon, confidence 0.73, low cost
and effort, sourced to FAO conservation agriculture guidance. It ranks first because it is
the only low-cost card that moves the carbon side and the moisture side at once, which is
what the diagnosis calls for.

Contour bunding and residue retention follow, and the ranking always closes with the
baseline monitoring card, because without fixed sampling points and a dated photo record
none of the above can be distinguished from a wet year.

---

## 7. Testing and CI

Ten tests, aimed at the places where a wrong answer is a scientific error rather than a
crash: condition gating in both directions, the two-variable requirement on diagnosis
rules, every recommendation carrying a source and a horizon, the three-variable minimum,
and the free-text extractor not assigning "low" to both halves of "rainfall is low but
pollution is high".

GitHub Actions runs `ruff`, `black --check`, `pytest`, and then executes the worked example
as a smoke test. CI installs only the light dependency set on purpose: the fallback paths
are the ones every reviewer will hit, so those are what get tested on every push.

---

## 8. Known limitations

- The knowledge base is hand-curated at assessment level. Real depth needs a paper corpus
  chunked and indexed with per-passage citation, which changes the retrieval problem
  from 26 cards to tens of thousands of passages.
- Session memory is in-process, so it is lost on restart and does not survive across
  multiple API replicas.
- Diagnosis rules are hand-written. They are explainable and testable, which is why they
  were chosen, but they cover the constraint combinations I could ground properly rather
  than the full space.
- Geo-inference from latitude is crude. Real spatial context needs land cover raster
  lookup, not a latitude band.
- Magnitude estimates are ranges from the literature, not site-specific predictions. The
  system does not model the site; it retrieves what has been observed in comparable ones.
- No authentication or per-user persistence. Not needed for the challenge, needed for
  anything real.
