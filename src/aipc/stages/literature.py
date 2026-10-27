"""Literature agent: OpenAlex search (retrieved) plus the model's own
parametric knowledge (SEP-level orientation) -> a LiteraturePack.

PhilArchive's OAI-PMH endpoint is blocked by a Cloudflare JS challenge for
any scripted client, and Semantic Scholar's unauthenticated tier is rate
limited, so OpenAlex's free-text search is the only retrieval route this
agent uses for now. The model first turns the topic into a few keyword
queries. Known coverage skew: OpenAlex indexes journal metadata, not full
text, and leans open-access/recent - it will systematically under-know
canonical book-length arguments.
"""

import os
import re
import time

import httpx

from aipc.context import Context
from aipc.models import LiteraturePack
from aipc.replies import list_items

QUOTED_TITLE = re.compile(r"[\"“]([^\"“”]{10,})[\"”]")
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_RETRIES = 2
OPENALEX_RETRY_SECONDS = 2
SEARCH_QUERIES = 3
RESULTS_PER_QUERY = 5


def search_openalex(query: str, limit: int = 8) -> list[dict]:
    """Free-text search. `?` and `*` are dropped: OpenAlex reads them as
    wildcards and rejects them in a normal search (HTTP 400 on a topic
    phrased as a question).

    Without a key, every caller on the network's IP address shares $0.10
    of searches a day at 10 a second; a key in OPENALEX_API_KEY has its
    own budget. It goes in a header, not the URL, which httpx logs. A 429
    is retried after a pause, as OpenAlex advises for the per-second
    limit; a spent daily budget still fails.
    """
    query = query.replace("?", " ").replace("*", " ")
    key = os.environ.get("OPENALEX_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    params = {"search": query, "per-page": limit}
    response = httpx.get(OPENALEX_WORKS_URL, params=params, headers=headers, timeout=30)
    for _ in range(OPENALEX_RETRIES):
        if response.status_code != 429:
            break
        time.sleep(OPENALEX_RETRY_SECONDS)
        response = httpx.get(OPENALEX_WORKS_URL, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()["results"]


def _format_source(work: dict) -> str:
    """Title, authors, year and, where OpenAlex has one, venue: the drafter
    writes its references from these, and without the venue it borrowed
    one from elsewhere."""
    authors = ", ".join(a["author"]["display_name"] for a in work.get("authorships", [])[:3])
    year = work.get("publication_year")
    venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
    return f"{work['display_name']} - {authors} ({year})" + (f", {venue}" if venue else "")


def search_queries(topic: str, ctx: Context) -> list[str]:
    """Short keyword queries written by the model: a sentence-length topic
    returns nothing from OpenAlex, and writing the
    queries by hand would add a human input beyond the topic.
    """
    reply = ctx.ask("literature_queries", topic=topic, n=SEARCH_QUERIES).text
    return list_items(reply)[:SEARCH_QUERIES]


def named_titles(topic: str) -> list[str]:
    """Titles the topic quotes, searched as they stand. A model's query
    mixing a named work's title with its authors' names returned unrelated
    papers; the bare title found the work first.
    """
    return QUOTED_TITLE.findall(topic)


def search_all(queries: list[str]) -> list[dict]:
    """Results of every query, first occurrence of each work kept."""
    works: dict[str, dict] = {}
    for query in queries:
        for work in search_openalex(query, limit=RESULTS_PER_QUERY):
            works.setdefault(work["id"], work)
    return list(works.values())


def run(topic: str, ctx: Context) -> LiteraturePack:
    """Works the topic names by title come first among the sources."""
    queries = named_titles(topic) + search_queries(topic, ctx)
    # A dry run skips the HTTP calls too: OpenAlex is free, but a dry run
    # should not depend on the network.
    works = [] if ctx.config.dry_run else search_all(queries)
    sources = [_format_source(work) for work in works]

    result = ctx.ask("literature_background", topic=topic, sources="\n".join(sources) or "(no results)")

    background_part, _, engage_part = result.text.partition("ENGAGE:")
    background = background_part.replace("BACKGROUND:", "").strip()
    engagement_list = [line.lstrip("- ").strip() for line in engage_part.strip().splitlines() if line.strip()]

    return LiteraturePack(background=background, queries=queries, sources=sources, engagement_list=engagement_list)
