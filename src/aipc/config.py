"""Pydantic config models, loaded from YAML and overridable by CLI flags."""

from pathlib import Path

import yaml
from pydantic import BaseModel

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class ModelConfig(BaseModel):
    """`supports_seed` and `supports_temperature` mirror OpenRouter's
    supported_parameters for the model (check with `aipc prices`): OpenRouter
    silently drops an unsupported parameter, so it is not sent at all.
    """

    id: str
    usd_per_mtok_in: float
    usd_per_mtok_out: float
    reasoning: bool = True
    supports_seed: bool = True
    supports_temperature: bool = True


class Defaults(BaseModel):
    tier: str
    temperature: float
    seed: int


class Limits(BaseModel):
    max_cost_usd: float
    target_words: tuple[int, int]


class PipelineConfig(BaseModel):
    strategy: str
    critics: int
    max_revision_rounds: int
    ideation_candidates: int
    decomposition_candidates: int
    debate_min_rounds: int
    debate_max_rounds: int


class Config(BaseModel):
    models: dict[str, ModelConfig]
    defaults: Defaults
    limits: Limits
    pipeline: PipelineConfig
    judge_prompt: str
    dry_run: bool = False

    def model_for_tier(self, tier: str) -> ModelConfig:
        return self.models[tier]


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)
