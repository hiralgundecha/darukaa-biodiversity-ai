"""API surface.

One conversational endpoint does the work. The others exist so a reviewer can
inspect the knowledge layer directly rather than inferring it from chat output.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import conversation, narrate
from .reasoning import REQUIRED_MIN, _count_signals, missing_questions, normalise, recommend
from .retriever import get_store
from .schemas import ChatRequest, ChatResponse, SiteProfile

app = FastAPI(
    title="Darukaa Biodiversity Intelligence API",
    version="1.0.0",
    description="Retrieval-grounded advisory system for land and biodiversity decisions.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html"


@app.get("/")
def index():
    if FRONTEND.exists():
        return FileResponse(FRONTEND)
    return {"service": "darukaa-biodiversity-ai", "docs": "/docs"}


@app.get("/health")
def health():
    store = get_store()
    return {
        "status": "ok",
        "knowledge_cards": len(store.cards),
        "vector_backend": store.backend,
        "embedding_backend": store.embedder.backend,
        "llm_narration": bool(os.getenv("ANTHROPIC_API_KEY")),
    }


@app.get("/knowledge")
def knowledge(limit: int = 50):
    store = get_store()
    return [
        {"id": c["id"], "practice": c["practice"], "source": c["source"], "evidence_tier": c["evidence_tier"]}
        for c in store.cards[:limit]
    ]


@app.get("/knowledge/{card_id}")
def knowledge_card(card_id: str):
    store = get_store()
    if card_id not in store.by_id:
        raise HTTPException(404, "no such card")
    return store.by_id[card_id]


@app.get("/search")
def search(q: str, k: int = 5):
    """Raw retrieval, no reasoning. Useful for checking what the index does."""
    return [{"id": c["id"], "practice": c["practice"], "score": round(s, 3)} for c, s in get_store().search(q, k=k)]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session = conversation.get(req.session_id)

    if req.site:
        session.merge(req.site.model_dump())
    if req.message:
        session.turns.append({"role": "user", "content": req.message})
        session.merge(conversation.extract_from_text(req.message))

    site = normalise(session.site)
    session.site = site
    signals = _count_signals(site)

    if signals < REQUIRED_MIN:
        fields, questions = missing_questions(site, limit=3)
        session.asked.extend(fields)
        conversation.save(session)
        known = site.known_fields()
        preamble = f"I have {', '.join(known)} so far. " if known else "I do not have anything about the site yet. "
        reply = (
            preamble + f"I need at least {REQUIRED_MIN} environmental variables before a recommendation is worth "
            "anything, because single-variable advice is where this goes wrong. Could you tell me:"
        )
        return ChatResponse(
            session_id=session.session_id,
            status="needs_input",
            reply=reply,
            known=known,
            missing_required=fields,
            questions=questions,
        )

    result = recommend(site, req.message)
    session.site = result["site"]
    conversation.save(session)

    text = narrate.template_reply(result["diagnoses"], result["recommendations"], result["signals"])
    source = "template"
    if os.getenv("ANTHROPIC_API_KEY"):
        generated = narrate.llm_reply(result["diagnoses"], result["recommendations"])
        if generated:
            text, source = generated, "llm"

    _, follow_ups = missing_questions(result["site"], limit=2)

    return ChatResponse(
        session_id=session.session_id,
        status="answered",
        reply=text,
        known=result["site"].known_fields(),
        questions=follow_ups,
        diagnoses=result["diagnoses"],
        recommendations=result["recommendations"],
        retrieved_ids=result["retrieved_ids"],
        narration_source=source,
    )


@app.post("/analyse", response_model=ChatResponse)
def analyse(site: SiteProfile) -> ChatResponse:
    """Stateless structured-input path, for scripted or batch use."""
    return chat(ChatRequest(site=site, message=None))


@app.delete("/session/{session_id}")
def clear(session_id: str):
    conversation.reset(session_id)
    return {"cleared": session_id}
