"""Per-run execution context passed to every stage, so stages take no
globals and a fake LLM client can be injected in tests.
"""

from dataclasses import dataclass

from aipc.config import Config
from aipc.llm import CallResult, LLMClient
from aipc.prompt_loader import load_prompt
from aipc.store import Store


@dataclass
class Context:
    llm: LLMClient
    config: Config
    run_id: str
    store: Store

    def render(self, prompt_name: str, **fields) -> str:
        """Fills in prompts/<prompt_name>.md, recording the template's hash
        so every run can name the exact prompt text it used.

        Writes to SQLite, so call it on the main thread.
        """
        template, prompt_hash = load_prompt(prompt_name)
        self.store.save_prompt(prompt_hash, agent_type=prompt_name, version="v1", text=template)
        return template.format(**fields)

    def ask(self, prompt_name: str, tier: str | None = None, seed: int | None = None, **fields) -> CallResult:
        """Renders a prompt and sends it, at the run's generator tier unless
        another is given.
        """
        prompt = self.render(prompt_name, **fields)
        return self.llm.call(tier=tier or self.config.defaults.tier, prompt=prompt, seed=seed, prompt_name=prompt_name)
