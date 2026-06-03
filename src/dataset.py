"""Prompt generation for indirect object identification experiments."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "prompts" / "ioi_templates.jsonl"

DEFAULT_NAMES = (
    "John",
    "Mary",
    "Alice",
    "Bob",
    "Tom",
    "Sarah",
    "Emma",
    "Jack",
    "Emily",
    "Daniel",
)

DEFAULT_PLACES = (
    "store",
    "school",
    "office",
    "park",
    "restaurant",
    "library",
)

DEFAULT_OBJECTS = (
    "book",
    "letter",
    "drink",
    "snack",
    "ticket",
    "phone",
)

FALLBACK_TEMPLATES = (
    "When {io} and {s} went to the {place}, {s} gave a {object} to",
    "After {io} and {s} arrived at the {place}, {s} handed a {object} to",
    "Then, {s} and {io} went to the {place}. {s} gave a {object} to",
)


@dataclass(frozen=True)
class IOIExample:
    """A paired clean/corrupted IOI prompt.

    In the clean prompt, `io_name` is the correct next-token answer and `s_name`
    is the subject distractor. The corrupted prompt swaps the two names.
    """

    clean_prompt: str
    corrupted_prompt: str
    io_name: str
    s_name: str
    place: str
    object_name: str
    template_id: str


def load_templates(path: str | Path = DEFAULT_TEMPLATE_PATH) -> list[dict[str, str]]:
    """Load JSONL prompt templates, falling back to built-ins if absent."""

    template_path = Path(path)
    if not template_path.exists():
        return [
            {"id": f"fallback_{idx}", "template": template}
            for idx, template in enumerate(FALLBACK_TEMPLATES)
        ]

    templates: list[dict[str, str]] = []
    with template_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "template" not in row:
                raise ValueError(f"Missing template in {template_path}:{line_number}")
            templates.append(
                {
                    "id": str(row.get("id", f"template_{line_number}")),
                    "template": str(row["template"]),
                }
            )

    if not templates:
        raise ValueError(f"No templates found in {template_path}")
    return templates


def build_ioi_dataset(
    n_examples: int,
    *,
    seed: int = 0,
    names: Iterable[str] = DEFAULT_NAMES,
    places: Iterable[str] = DEFAULT_PLACES,
    objects: Iterable[str] = DEFAULT_OBJECTS,
    template_path: str | Path = DEFAULT_TEMPLATE_PATH,
    template_id: str | None = None,
) -> list[IOIExample]:
    """Build clean/corrupted IOI prompt pairs.

    The prompts end immediately before the indirect object answer token. The
    model should prefer `io_name` over `s_name` on the clean prompt.
    """

    if n_examples < 1:
        raise ValueError("n_examples must be positive")

    rng = random.Random(seed)
    name_pool = tuple(names)
    place_pool = tuple(places)
    object_pool = tuple(objects)
    templates = load_templates(template_path)
    if template_id is not None:
        templates = [template for template in templates if template["id"] == template_id]
        if not templates:
            raise ValueError(f"Template id not found: {template_id}")

    if len(name_pool) < 2:
        raise ValueError("At least two names are required")
    if not place_pool:
        raise ValueError("At least one place is required")
    if not object_pool:
        raise ValueError("At least one object is required")

    examples: list[IOIExample] = []
    for _ in range(n_examples):
        io_name, s_name = rng.sample(name_pool, 2)
        place = rng.choice(place_pool)
        object_name = rng.choice(object_pool)
        template = rng.choice(templates)

        clean_prompt = template["template"].format(
            io=io_name,
            s=s_name,
            place=place,
            object=object_name,
        )
        corrupted_prompt = template["template"].format(
            io=s_name,
            s=io_name,
            place=place,
            object=object_name,
        )

        examples.append(
            IOIExample(
                clean_prompt=clean_prompt,
                corrupted_prompt=corrupted_prompt,
                io_name=io_name,
                s_name=s_name,
                place=place,
                object_name=object_name,
                template_id=template["id"],
            )
        )

    return examples


def clean_prompts(examples: Iterable[IOIExample]) -> list[str]:
    """Return clean prompt strings."""

    return [example.clean_prompt for example in examples]


def corrupted_prompts(examples: Iterable[IOIExample]) -> list[str]:
    """Return corrupted prompt strings."""

    return [example.corrupted_prompt for example in examples]
