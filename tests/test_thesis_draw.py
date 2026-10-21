import pytest

from aipc.models import FilterVerdict, ScreenedThesis, ThesisCandidate
from aipc.stages import thesis_draw

PASS = FilterVerdict(passed=True, reasoning="")
FAIL = FilterVerdict(passed=False, reasoning="")


def _write_candidates(run_dir, specs):
    run_dir.mkdir(exist_ok=True)
    for i, (claim, significance, novelty) in enumerate(specs):
        screened = ScreenedThesis(
            candidate=ThesisCandidate(claim=claim, origin="cold", rationale=""),
            significance=significance,
            novelty=novelty,
        )
        (run_dir / f"candidate{i}.json").write_text(screened.model_dump_json())


def test_load_survivors_keeps_only_double_passes_in_order(tmp_path):
    specs = [("one", PASS, PASS), ("two", FAIL, None), ("three", PASS, FAIL)] + [(f"c{i}", PASS, PASS) for i in range(8)]
    _write_candidates(tmp_path, specs)
    assert thesis_draw.load_survivors(tmp_path) == ["one"] + [f"c{i}" for i in range(8)]


def test_draw_is_deterministic_for_a_seed():
    survivors = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]
    assert thesis_draw.draw(survivors, 2, seed=3) == thesis_draw.draw(survivors, 2, seed=3)


def test_draw_rejects_near_duplicates_and_records_the_seed():
    survivors = ["free will survives determinism", "free will survives determinism arguments"]
    with pytest.raises(ValueError, match="near-duplicate"):
        thesis_draw.draw(survivors, 2, seed=1)

    survivors.append("consciousness is not computational")
    result = thesis_draw.draw(survivors, 2, seed=1)
    assert thesis_draw.overlap(*result.theses) <= thesis_draw.MAX_OVERLAP
    assert result.seed == 1 + len(result.rejected_seeds)


def test_draw_needs_enough_survivors():
    with pytest.raises(ValueError, match="only 1 survivors"):
        thesis_draw.draw(["only one"], 2, seed=1)
