import pytest

from aipc.replies import is_none, list_items, parse_entries, parse_verdict, parse_winner


def test_parse_verdict_reads_the_verdict_line():
    assert parse_verdict("VERDICT: PASS\nREASONING:\nFine.").passed
    assert not parse_verdict("VERDICT: FAIL\nREASONING:\nBroken.").passed
    assert not parse_verdict("**VERDICT:** FAIL\n**REASONING:** x").passed
    assert parse_verdict("Verdict: pass\nREASONING: fine").passed


def test_parse_verdict_never_reads_fail_as_pass():
    assert not parse_verdict("VERDICT: FAIL - it does not pass\nREASONING: x").passed
    assert not parse_verdict("VERDICT: FAIL\nThe thesis does not surpass existing work.").passed
    assert not parse_verdict("VERDICT: PASS or FAIL: FAIL\nREASONING: x").passed


def test_parse_verdict_raises_on_an_unreadable_reply():
    for reply in ("I think this passes.", "VERDICT: PASS or FAIL"):
        with pytest.raises(ValueError, match="unreadable verdict"):
            parse_verdict(reply)


def test_a_dry_run_fails_an_unreadable_verdict_and_keeps_it():
    verdict = parse_verdict("[dry-run: no call made]", dry_run=True)
    assert not verdict.passed
    assert verdict.reasoning == "[dry-run: no call made]"


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


LEDGER_KEYS = {"CLAIM", "SUPPORT", "OBJECTION", "REPLY", "REPLY_TYPE", "CONCESSION"}


@pytest.mark.parametrize(
    "reply",
    [
        "CLAIM: c\nSUPPORT: s\nOBJECTION: o\nREPLY: r\nREPLY_TYPE: decisive",
        "**CLAIM:** c\n**SUPPORT:** s\n**OBJECTION:** o\n**REPLY:** r\n**REPLY_TYPE:** decisive",
        "**CLAIM**: c\n**SUPPORT**: s\n**OBJECTION**: o\n**REPLY**: r\n**REPLY_TYPE**: decisive",
        "- CLAIM: c\n- SUPPORT: s\n- OBJECTION: o\n- REPLY: r\n- REPLY_TYPE: decisive",
        "1. CLAIM 1: c\nSUPPORT: s\n\nOBJECTION: o\nREPLY: r\nREPLY TYPE: decisive",
        "Here is the ledger entry:\n\nCLAIM: c\nSUPPORT: s\nOBJECTION: o\nREPLY: r\nREPLY_TYPE: decisive",
    ],
)
def test_parse_entries_reads_decorated_keys_and_blank_lines_inside_an_entry(reply):
    assert parse_entries(reply, LEDGER_KEYS, start="CLAIM") == [
        {"CLAIM": "c", "SUPPORT": "s", "OBJECTION": "o", "REPLY": "r", "REPLY_TYPE": "decisive"}
    ]


def test_parse_entries_joins_wrapped_values_and_splits_at_each_start_key():
    reply = (
        "CLAIM: first\nSUPPORT: Frankfurt (1971): the principle\nof alternate possibilities.\n\n"
        "CLAIM: second\nSUPPORT: s2\n\nCONCESSION: k1\nCONCESSION: k2\n\nThese claims together answer the question."
    )
    assert parse_entries(reply, LEDGER_KEYS, start="CLAIM") == [
        {"CLAIM": "first", "SUPPORT": "Frankfurt (1971): the principle of alternate possibilities."},
        {"CLAIM": "second", "SUPPORT": "s2", "CONCESSION": "k1"},
        {"CONCESSION": "k2"},
    ]


def test_is_none_reads_only_the_first_line():
    assert is_none("NONE\n\nThe latest round only restated earlier points.")
    assert not is_none("The essay never answers the regress worry.\n\nNONE of its replies address it.")


def test_list_items_drops_markers_quotes_and_an_introducing_line():
    reply = (
        'Here are the sub-questions:\n\n1. What is X?\n- Does Y follow?\n**3.** Is Z true?\n"quoted query"\n* starred'
    )
    assert list_items(reply) == ["What is X?", "Does Y follow?", "Is Z true?", "quoted query", "starred"]
