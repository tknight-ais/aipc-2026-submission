from aipc.config import DEFAULT_CONFIG_PATH, load_config


def test_load_config_resolves_tier():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.model_for_tier("cheap").id == "anthropic/claude-haiku-5.5"
    assert config.defaults.tier == "cheap"
    assert config.limits.target_words == (3000, 5000)
