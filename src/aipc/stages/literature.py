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

import httpx

from aipc.context import Context
from aipc.models import LiteraturePack
from aipc.replies import list_items

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SEARCH_QUERIES = 3
RESULTS_PER_QUERY = 5


def search_openalex(query: str, limit: int = 8) -> list[dict]:
    """Free-text search. `?` and `*` are dropped: OpenAlex reads them as
    wildcards and rejects them in a normal search (HTTP 400 on a topic
    phrased as a question).
    """
    query = query.replace("?", " ").replace("*", " ")
    response = httpx.get(OPENALEX_WORKS_URL, params={"search": query, "per-page": limit}, timeout=30)
    response.raise_for_status()
    return response.json()["results"]


def _format_source(work: dict) -> str:
    authors = ", ".join(a["author"]["display_name"] for a in work.get("authorships", [])[:3])
    year = work.get("publication_year")
    return f"{work['display_name']} - {authors} ({year})"


def search_queries(topic: str, ctx: Context) -> list[str]:
    """Short keyword queries written by the model: a sentence-length topic
    returns nothing from OpenAlex, and writing the
    queries by hand would add a human input beyond the topic.
    """
    reply = ctx.ask("literature_queries", topic=topic, n=SEARCH_QUERIES).text
    return list_items(reply)[:SEARCH_QUERIES]


def search_all(queries: list[str]) -> list[dict]:
    """Results of every query, first occurrence of each work kept."""
    works: dict[str, dict] = {}
    for query in queries:
        for work in search_openalex(query, limit=RESULTS_PER_QUERY):
            works.setdefault(work["id"], work)
    return list(works.values())


def run(topic: str, ctx: Context) -> LiteraturePack:
    queries = search_queries(topic, ctx)
    # A dry run skips the HTTP calls too: OpenAlex is free, but a dry run
    # should not depend on the network.
    works = [] if ctx.config.dry_run else search_all(queries)
    sources = [_format_source(work) for work in works]

    result = ctx.ask("literature_background", topic=topic, sources="\n".join(sources) or "(no results)")

    background_part, _, engage_part = result.text.partition("ENGAGE:")
    background = background_part.replace("BACKGROUND:", "").strip()
    engagement_list = [line.lstrip("- ").strip() for line in engage_part.strip().splitlines() if line.strip()]

    return LiteraturePack(background=background, queries=queries, sources=sources, engagement_list=engagement_list)
