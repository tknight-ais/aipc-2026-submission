from aipc.config import ModelConfig
from aipc.prices import check


def _live(model_id, price_in, price_out, parameters):
    return {
        "id": model_id,
        "pricing": {"prompt": str(price_in / 1_000_000), "completion": str(price_out / 1_000_000)},
        "supported_parameters": parameters,
    }


def test_matching_entry_is_ok():
    models = {"judge": ModelConfig(id="g/flash", usd_per_mtok_in=0.3, usd_per_mtok_out=2.5)}
    assert check(models, [_live("g/flash", 0.3, 2.5, ["seed", "temperature"])]) == ["ok judge: g/flash"]


def test_stale_price_and_unsupported_parameters_are_flagged():
    models = {"judge": ModelConfig(id="o/sol", usd_per_mtok_in=2.0, usd_per_mtok_out=10.0)}
    [line] = check(models, [_live("o/sol", 4.0, 20.0, ["seed"])])
    assert line.startswith("MISMATCH judge")
    assert "$4.00/$20.00 live" in line
    assert "supports_temperature is True in config, False live" in line


def test_unlisted_model_is_flagged():
    models = {"gone": ModelConfig(id="x/old", usd_per_mtok_in=1.0, usd_per_mtok_out=1.0)}
    assert check(models, []) == ["MISMATCH gone: x/old is not listed"]
