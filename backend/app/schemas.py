"""Request and response shapes for the API."""

from typing import Any, Literal

from pydantic import BaseModel, Field

Horizon = Literal["short", "medium", "long"]


class SiteProfile(BaseModel):
    """Everything the system knows about one piece of land.

    Every field is optional on purpose. The whole point of the conversation
    layer is that a user arrives with two facts and we work out which of the
    missing ones actually change the answer.
    """

    soil_organic_carbon_pct: float | None = Field(None, ge=0, le=25)
    soil_ph: float | None = Field(None, ge=2, le=12)
    soil_moisture: Literal["low", "moderate", "high"] | None = None
    land_use: str | None = None
    climate: str | None = None
    rainfall: Literal["low", "moderate", "high"] | None = None
    rainfall_mm: float | None = Field(None, ge=0)
    temperature_c: float | None = None
    species_richness: int | None = Field(None, ge=0)
    habitat_fragmentation: Literal["low", "moderate", "high"] | None = None
    pollution: Literal["low", "moderate", "high"] | None = None
    deforestation: Literal["low", "moderate", "high"] | None = None
    slope: Literal["flat", "moderate", "steep"] | None = None
    area_ha: float | None = Field(None, gt=0)
    region: str | None = None
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)

    def known_fields(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str | None = None
    site: SiteProfile | None = None


class Evidence(BaseModel):
    kb_id: str
    source: str
    evidence_tier: int


class MetricEffect(BaseModel):
    metric: str
    direction: Literal["increase", "decrease"]
    magnitude: str
    horizon: Horizon


class Recommendation(BaseModel):
    rank: int
    action: str
    detail: str
    why_it_works: str
    addresses: list[str]
    metrics: list[MetricEffect]
    linked_variables: list[str]
    horizon: Horizon
    confidence: float
    confidence_label: Literal["low", "moderate", "high"]
    cost: str
    effort: str
    evidence: Evidence
    retrieval_score: float


class Diagnosis(BaseModel):
    code: str
    label: str
    explanation: str
    variables: list[str]


class ChatResponse(BaseModel):
    session_id: str
    status: Literal["needs_input", "answered"]
    reply: str
    known: dict[str, Any] = {}
    missing_required: list[str] = []
    questions: list[str] = []
    diagnoses: list[Diagnosis] = []
    recommendations: list[Recommendation] = []
    retrieved_ids: list[str] = []
    narration_source: Literal["template", "llm"] = "template"
