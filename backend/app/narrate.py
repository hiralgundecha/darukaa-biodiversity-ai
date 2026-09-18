"""Turning the structured result into prose.

The template path is the default and is what runs in CI. An LLM is optional and
is only ever given the already-decided recommendations to phrase, never the
raw question. That ordering is the point: the model cannot invent a practice,
a number or a citation, because it does not choose any of them. If the API key
is absent or the call fails, the template output is returned unchanged.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .schemas import Diagnosis, Recommendation

SYSTEM_PROMPT = (
    "You are writing the covering explanation for an environmental advisory system. "
    "You will be given a diagnosis and a ranked list of recommendations that have already "
    "been selected from a vetted knowledge base. Rephrase them into three or four short "
    "paragraphs for a land manager. Do not add any practice, figure, citation or claim that "
    "is not present in the input. Do not soften the reasoning into generic advice."
)


def template_reply(diagnoses: list[Diagnosis], recs: list[Recommendation], signals: int) -> str:
    lines: list[str] = []
    if diagnoses:
        lines.append("Reading the variables together rather than separately:")
        for d in diagnoses:
            lines.append(f"- {d.label}. {d.explanation}")
    else:
        lines.append(
            "No compound constraint stood out from the values given, so the ranking below is driven by land use and general condition."
        )

    lines.append("")
    lines.append("Recommendations, ordered by how much of the diagnosed problem they address:")
    for r in recs:
        metrics = ", ".join(f"{m.metric.replace('_', ' ')} {m.direction}" for m in r.metrics)
        lines.append("")
        lines.append(f"{r.rank}. {r.action}")
        lines.append(f"   What it is: {r.detail}")
        lines.append(f"   Why it works: {r.why_it_works}")
        lines.append(f"   Metrics affected: {metrics}")
        if r.linked_variables:
            lines.append(f"   Variables connected: {'; '.join(r.linked_variables)}")
        lines.append(
            f"   Time horizon: {r.horizon} term | Confidence: {r.confidence_label} ({r.confidence}) | Cost: {r.cost}, effort: {r.effort}"
        )
        lines.append(f"   Evidence: {r.evidence.source}")

    if signals < 5:
        lines.append("")
        lines.append(
            "Confidence is capped because the profile is still thin. Adding pH, fragmentation and pollution level would narrow the ranking further."
        )
    return "\n".join(lines)


def llm_reply(diagnoses: list[Diagnosis], recs: list[Recommendation]) -> str | None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic

        payload: dict[str, Any] = {
            "diagnoses": [d.model_dump() for d in diagnoses],
            "recommendations": [r.model_dump() for r in recs],
        }
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text").strip() or None
    except Exception:
        return None
