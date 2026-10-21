"""Compares config model entries with OpenRouter's live model listing.

Stale prices make the dry run and the cost ceiling wrong (one judge was
once configured at half its price), and a parameter the model doesn't
support is silently dropped. Run `aipc prices` before any paid step.
"""

from aipc.config import ModelConfig

SAMPLING_FLAGS = {"supports_seed": "seed", "supports_temperature": "temperature"}


def _problems(model: ModelConfig, live: dict) -> list[str]:
    live_in = float(live["pricing"]["prompt"]) * 1_000_000
    live_out = float(live["pricing"]["completion"]) * 1_000_000
    problems = []
    if (round(live_in, 4), round(live_out, 4)) != (model.usd_per_mtok_in, model.usd_per_mtok_out):
        problems.append(f"price ${model.usd_per_mtok_in}/${model.usd_per_mtok_out} in config, ${live_in:.2f}/${live_out:.2f} live")
    supported = set(live.get("supported_parameters", []))
    for flag, parameter in SAMPLING_FLAGS.items():
        if getattr(model, flag) != (parameter in supported):
            problems.append(f"{flag} is {getattr(model, flag)} in config, {parameter in supported} live")
    return problems


def check(models: dict[str, ModelConfig], listing: list[dict]) -> list[str]:
    """One line per config entry; a line that needs action starts with MISMATCH."""
    by_id = {m["id"]: m for m in listing}
    lines = []
    for name, model in models.items():
        if model.id not in by_id:
            lines.append(f"MISMATCH {name}: {model.id} is not listed")
            continue
        problems = _problems(model, by_id[model.id])
        lines.append(f"MISMATCH {name}: {model.id}: {'; '.join(problems)}" if problems else f"ok {name}: {model.id}")
    return lines
