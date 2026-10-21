"""Local HTTP server for browsing runs. Bound to localhost only.

Read-only except /rate, which records blind human ratings (see
rating.py). URLs carry a run label that is looked up in the database;
nothing from the URL or a form is ever used as a file path.
"""

from contextlib import closing
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote

from aipc.store import Store
from aipc.viewer import data, rating, render


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, runs_dir: Path, experiment: str, **kwargs):
        self.runs_dir = runs_dir
        self.experiment = experiment
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        html = self.route(self.path)
        if html is None:
            self.send_error(404)
            return
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/rate":
            self.send_error(404)
            return
        form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
        store = Store(self.runs_dir)
        with closing(store.conn):
            rating.record(store, self.experiment, int(form["pair"][0]), form["choice"][0])
        self.send_response(303)
        self.send_header("Location", "/rate")
        self.end_headers()

    def rate_page(self) -> str:
        store = Store(self.runs_dir)
        with closing(store.conn):
            total = len(rating.anchors(store, self.experiment))
            pending = rating.next_unrated(store, self.experiment)
            if pending is None:
                return render.rate_done_page(total)
            index, anchor = pending
            first = rating.essay_text(store, self.runs_dir, anchor.shown_first_run)
            second = rating.essay_text(store, self.runs_dir, anchor.shown_second_run)
        return render.rate_page(index, total, first, second)

    def route(self, path: str) -> str | None:
        if path == "/rate":
            return self.rate_page()
        if path == "/":
            return render.index_page(data.list_runs(self.runs_dir))
        if path.startswith("/run/"):
            run = data.get_run(self.runs_dir, unquote(path.removeprefix("/run/")))
            if run is None:
                return None
            config = data.get_config(self.runs_dir, run["config_hash"])
            return render.run_page(run, config, data.get_stages(self.runs_dir, run))
        return None


def make_server(runs_dir: Path, port: int, experiment: str = "poc") -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, runs_dir=runs_dir, experiment=experiment))
