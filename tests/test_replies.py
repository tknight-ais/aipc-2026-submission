from aipc.replies import is_none, parse_verdict, parse_winner


def test_parse_verdict_reads_the_verdict_line():
    assert parse_verdict("VERDICT: PASS\nREASONING:\nFine.").passed
    assert not parse_verdict("VERDICT: FAIL\nREASONING:\nBroken.").passed
    assert not parse_verdict("**VERDICT:** FAIL\n**REASONING:** x").passed
    assert parse_verdict("Verdict: pass\nREASONING: fine").passed


def test_parse_verdict_never_reads_fail_as_pass():
    assert not parse_verdict("VERDICT: FAIL - it does not pass\nREASONING: x").passed
    assert not parse_verdict("VERDICT: FAIL\nThe thesis does not surpass existing work.").passed
    assert not parse_verdict("VERDICT: PASS or FAIL: FAIL\nREASONING: x").passed


def test_parse_verdict_fails_an_unreadable_reply_and_keeps_it():
    verdict = parse_verdict("I think this passes.")
    assert not verdict.passed
    assert verdict.reasoning == "I think this passes."


def test_parse_winner_reads_essay_prefix_and_markdown():
    assert parse_winner("REASONING: ...\nWINNER: Essay A") == "A"
    assert parse_winner("REASONING: ...\nWINNER: **B**") == "B"
    assert parse_winner("REASONING: ...\nWinner - B.") == "B"


def test_parse_winner_ignores_a_lower_case_article():
    assert parse_winner("REASONING:\nWinner: a close call, but I cannot decide") is None


def test_is_none_tolerates_markdown_and_a_full_stop():
    assert is_none("NONE")
    assert is_none("NONE.")
    assert is_none("**NONE**")
    assert is_none(" none \n")
    assert not is_none("None of the objections above is answered.")
