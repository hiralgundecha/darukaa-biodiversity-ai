"""Build the index and run the brief's worked example end to end.

Run this first. It tells you which backends actually came up, which is more
useful than a silent success when an optional dependency is missing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.narrate import template_reply  # noqa: E402
from app.reasoning import recommend  # noqa: E402
from app.retriever import get_store  # noqa: E402
from app.schemas import SiteProfile  # noqa: E402

EXAMPLE = SiteProfile(
    soil_organic_carbon_pct=0.3,
    rainfall="low",
    land_use="monoculture",
    climate="semi-arid",
    region="semi-arid India",
)


def main() -> None:
    store = get_store()
    print(f"knowledge cards indexed : {len(store.cards)}")
    print(f"vector backend          : {store.backend}")
    print(f"embedding backend       : {store.embedder.backend}")
    print("-" * 72)

    result = recommend(EXAMPLE, "biodiversity is declining on my land")
    print(template_reply(result["diagnoses"], result["recommendations"], result["signals"]))


if __name__ == "__main__":
    main()
