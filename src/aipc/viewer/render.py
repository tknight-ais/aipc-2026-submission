"""HTML rendering for the viewer. Plain strings, everything escaped."""

from html import escape
from urllib.parse import quote

STYLE = """
:root { --bg: #fff; --fg: #1d1d1f; --muted: #6e6e73; --line: #d8d8dc; --card: #f5f5f7; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #1c1c1e; --fg: #f2f2f7; --muted: #98989d; --line: #3a3a3c; --card: #2c2c2e; }
}
body { font: 15px/1.5 system-ui, sans-serif; background: var(--bg); color: var(--fg);
       max-width: 960px; margin: 0 auto; padding: 24px 16px; }
a { color: inherit; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 500; }
details { border: 1px solid var(--line); border-radius: 6px; margin: 10px 0; }
summary { cursor: pointer; padding: 8px 12px; background: var(--card); font-weight: 600; }
summary .type { color: var(--muted); font-weight: 400; margin-left: 8px; }
.body { padding: 4px 12px 12px; }
.key { color: var(--muted); font-size: 13px; margin-top: 8px; }
.text { white-space: pre-wrap; background: var(--card); padding: 8px 10px; border-radius: 4px; }
.muted { color: var(--muted); }
ol, ul { margin: 4px 0; padding-left: 22px; }
"""


def page(title: str, body: str) -> str:
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{escape(title)}</title>"
        f"<style>{STYLE}</style></head><body>{body}</body></html>"
    )


def render_value(value) -> str:
    """Generic recursive view of any JSON value."""
    if isinstance(value, dict):
        return "".join(
            f"<div class='key'>{escape(str(k))}</div>{render_value(v)}" for k, v in value.items()
        )
    if isinstance(value, list):
        if not value:
            return "<span class='muted'>(empty)</span>"
        return "<ol>" + "".join(f"<li>{render_value(v)}</li>" for v in value) + "</ol>"
    if isinstance(value, str) and ("\n" in value or len(value) > 80):
        return f"<div class='text'>{escape(value)}</div>"
    return escape(str(value))


def render_stage(stage: dict) -> str:
    opened = " open" if stage["stage"] == "essay" else ""
    return (
        f"<details{opened}><summary>{escape(stage['stage'])}"
        f"<span class='type'>{escape(stage['type'])}</span></summary>"
        f"<div class='body'>{render_value(stage['data'])}</div></details>"
    )


def index_page(runs: list[dict]) -> str:
    rows = "".join(
        f"<tr><td><a href='/run/{quote(r['label'])}'>{escape(r['label'])}</a></td>"
        f"<td>{escape(r['variant'])}</td><td>{escape(r['status'])}</td>"
        f"<td>${r['cost_usd']:.4f}</td><td>{r['word_count'] or ''}</td></tr>"
        for r in runs
    )
    table = (
        "<table><tr><th>Run</th><th>Variant</th><th>Status</th><th>Cost</th><th>Essay words</th></tr>"
        f"{rows}</table>"
    )
    return page("AIPC runs", f"<h1>AIPC runs</h1>{table}")


def run_page(run: dict, config: dict, stages: list[dict]) -> str:
    header = (
        f"<p><a href='/'>All runs</a></p><h1>{escape(run['label'])}</h1>"
        f"<p class='muted'>variant {escape(run['variant'])} - status {escape(run['status'])} - "
        f"cost ${run['cost_usd']:.4f} - config {escape(run['config_hash'][:12])}</p>"
    )
    config_block = (
        "<details><summary>resolved config</summary>"
        f"<div class='body'>{render_value(config)}</div></details>"
    )
    return page(run["label"], header + config_block + "".join(render_stage(s) for s in stages))


RATE_STYLE = """
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 800px) { .pair { grid-template-columns: 1fr; } }
.pair .text { max-height: 70vh; overflow-y: auto; }
form { display: flex; gap: 12px; margin: 16px 0; }
button { font: inherit; padding: 8px 16px; border-radius: 6px; border: 1px solid var(--line);
         background: var(--card); color: var(--fg); cursor: pointer; }
"""


def rate_page(index: int, total: int, first_text: str, second_text: str) -> str:
    """Anchors are rated in order, so `index` is also the number rated so far."""
    form = (
        f"<form method='post' action='/rate'><input type='hidden' name='pair' value='{index}'>"
        "<button name='choice' value='first'>First is better</button>"
        "<button name='choice' value='second'>Second is better</button></form>"
    )
    body = (
        f"<style>{RATE_STYLE}</style><h1>Blind rating</h1>"
        f"<p class='muted'>{index} of {total} rated. Which is the better piece of philosophy: "
        "a sounder, more rigorous argument, not better prose style.</p>"
        f"{form}<div class='pair'>"
        f"<div><h2>First</h2><div class='text'>{escape(first_text)}</div></div>"
        f"<div><h2>Second</h2><div class='text'>{escape(second_text)}</div></div></div>{form}"
    )
    return page("Blind rating", body)


def rate_done_page(total: int) -> str:
    if total == 0:
        return page(
            "Blind rating",
            "<h1>Blind rating</h1><p>No anchors drawn yet. Run the tournament first, then "
            "<code>aipc draw-anchors --experiment &lt;name&gt;</code>.</p>",
        )
    return page("Blind rating", f"<h1>Blind rating</h1><p>All {total} pairs rated. Thank you.</p>")
