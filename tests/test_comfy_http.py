"""ComfyGenerator against a minimal fake ComfyUI HTTP server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vspr.comfy import ComfyGenerator

REPO = Path(__file__).resolve().parent.parent


class FakeComfy(BaseHTTPRequestHandler):
    queued: list = []
    history: dict = {}

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/prompt":
            wf = payload["prompt"]
            pid = f"p{len(self.queued)}"
            self.queued.append(wf)
            batch = next(v["inputs"]["batch_size"] for v in wf.values() if "batch_size" in v["inputs"])
            save = next(k for k, v in wf.items() if v["class_type"] == "SaveImage")
            self.history[pid] = {"status": {"status_str": "success", "completed": True},
                                 "outputs": {save: {"images": [{"filename": f"{pid}_{b}.png", "subfolder": "vspr",
                                                                "type": "output"} for b in range(batch)]}}}
            self._send(json.dumps({"prompt_id": pid}).encode())
        else:  # /free
            self._send(b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path.startswith("/history/"):
            pid = url.path.rsplit("/", 1)[1]
            self._send(json.dumps({pid: self.history[pid]} if pid in self.history else {}).encode())
        elif url.path == "/view":
            self._send(f"PNG:{parse_qs(url.query)['filename'][0]}".encode(), "image/png")


def test_generate_fetch_and_cache(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        gen = ComfyGenerator(backend="sdxl", workflow_path=REPO / "workflows" / "sdxl.json",
                             cache_dir=tmp_path, url=f"http://127.0.0.1:{server.server_port}",
                             batch_size=4, poll_interval=0.01)
        files = gen.generate("a cat behind a dog", 7)
        assert len(files) == 4 and all(Path(f).exists() for f in files)
        assert Path(files[0]).read_bytes() == b"PNG:p0_0.png"
        assert FakeComfy.queued[0]["15"]["inputs"]["text"] == "a cat behind a dog"  # refiner prompt
        assert gen.generate("a cat behind a dog", 7) == files  # served from cache
        assert len(FakeComfy.queued) == 1
        gen.release()
    finally:
        server.shutdown()


def test_negative_prompt(tmp_path):
    from vspr.comfy import DEFAULT_NEGATIVE, _cache_key

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeComfy.queued, FakeComfy.history = [], {}
    try:
        gen = ComfyGenerator(backend="sdxl", workflow_path=REPO / "workflows" / "sdxl.json",
                             cache_dir=tmp_path, url=f"http://127.0.0.1:{server.server_port}",
                             batch_size=4, poll_interval=0.01)
        plain = gen.generate("a cat behind a dog", 7)
        assert [FakeComfy.queued[0][n]["inputs"]["text"] for n in ("7", "16")] == [DEFAULT_NEGATIVE] * 2
        neg = gen.generate("a cat behind a dog", 7, negative="cat on dog, text, watermark")
        assert len(FakeComfy.queued) == 2 and neg != plain  # a different negative is a different image set
        assert [FakeComfy.queued[1][n]["inputs"]["text"] for n in ("7", "16")] == ["cat on dog, text, watermark"] * 2
        assert FakeComfy.queued[1]["6"]["inputs"]["text"] == "a cat behind a dog"
        assert gen.generate("a cat behind a dog", 7, negative="cat on dog, text, watermark") == neg  # cached
        assert len(FakeComfy.queued) == 2
        # the default negative keeps the cache keys of images generated before negatives were supported
        wf = gen.workflow
        assert _cache_key("sdxl", "p", 1, 4, wf) == _cache_key("sdxl", "p", 1, 4, wf, DEFAULT_NEGATIVE)
        assert _cache_key("sdxl", "p", 1, 4, wf) != _cache_key("sdxl", "p", 1, 4, wf, "cow")
    finally:
        server.shutdown()


def test_cascade_negative_node():
    gen = ComfyGenerator(backend="cascade", workflow_path=REPO / "workflows" / "cascade.json", cache_dir=Path("."))
    wf = gen.build_workflow("a cat", 1, negative="dog")
    assert wf["7"]["inputs"]["text"] == "dog" and wf["6"]["inputs"]["text"] == "a cat"


def test_cfg(tmp_path):
    from vspr.comfy import _cache_key

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeComfy.queued, FakeComfy.history = [], {}
    try:
        gen = ComfyGenerator(backend="sdxl", workflow_path=REPO / "workflows" / "sdxl.json",
                             cache_dir=tmp_path, url=f"http://127.0.0.1:{server.server_port}",
                             batch_size=4, poll_interval=0.01)
        plain = gen.generate("a cat behind a dog", 7)
        assert [FakeComfy.queued[0][n]["inputs"]["cfg"] for n in ("10", "11")] == [8, 8]
        # the workflow's own value is the same request as no cfg (served from cache)
        assert gen.generate("a cat behind a dog", 7, cfg=8) == plain and len(FakeComfy.queued) == 1
        low = gen.generate("a cat behind a dog", 7, cfg=4)
        assert len(FakeComfy.queued) == 2 and low != plain
        assert [FakeComfy.queued[1][n]["inputs"]["cfg"] for n in ("10", "11")] == [4.0, 4.0]
        assert FakeComfy.queued[1]["10"]["inputs"]["noise_seed"] == 7
        assert gen.generate("a cat behind a dog", 7, cfg=4.0) == low and len(FakeComfy.queued) == 2
        both = gen.generate("a cat behind a dog", 7, negative="cat on dog", cfg=4)
        assert len(FakeComfy.queued) == 3 and both not in (plain, low)
        assert FakeComfy.queued[2]["7"]["inputs"]["text"] == "cat on dog"
        wf = gen.workflow
        assert _cache_key("sdxl", "p", 1, 4, wf) == _cache_key("sdxl", "p", 1, 4, wf, cfg=None)
        assert _cache_key("sdxl", "p", 1, 4, wf, cfg=4) != _cache_key("sdxl", "p", 1, 4, wf, cfg=11)
        assert _cache_key("sdxl", "p", 1, 4, wf, "cow", 4) != _cache_key("sdxl", "p", 1, 4, wf, cfg=4)
    finally:
        server.shutdown()


def test_cascade_cfg_node():
    gen = ComfyGenerator(backend="cascade", workflow_path=REPO / "workflows" / "cascade.json", cache_dir=Path("."))
    wf = gen.build_workflow("a cat", 1, cfg=6)
    assert wf["3"]["inputs"]["cfg"] == 6.0 and wf["33"]["inputs"]["cfg"] == 1.1  # stage B untouched
    assert gen._cfg(4) is None and gen._cfg(6) == 6.0
