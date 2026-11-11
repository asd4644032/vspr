"""ComfyUI API client implementing the frozen text-to-image model G.

Images are fetched back through the HTTP API and stored in a content-addressed
cache, so the result does not depend on where ComfyUI writes its own output and an
identical (backend, prompt, seed, batch) request is never generated twice. Round 1
of VSPR (P_1 = D) is therefore shared with the no-alignment baseline and with the
ablations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .io_utils import read_json, write_json

log = logging.getLogger(__name__)


class ImageGenerator(Protocol):
    name: str

    def generate(self, prompt: str, seed: int, negative: str = ..., cfg: float | None = ...) -> list[str]: ...

    def release(self) -> None: ...


# Negative prompt of the shipped workflows (the paper's setting).
DEFAULT_NEGATIVE = "text, watermark"

# Node ids of the workflows shipped in ``workflows/``.
NODE_MAPS = {
    "sdxl": {
        "workflow": "workflows/sdxl.json",
        # base and refiner both receive the prompt
        "positive": [["6", "text"], ["15", "text"]],
        "negative": [["7", "text"], ["16", "text"]],
        # refiner continues the base latent with add_noise=disable, so only the base seed matters
        "seed": [["10", "noise_seed"]],
        "batch": ["5", "batch_size"],
        # guidance scale of base and refiner (one value for both)
        "cfg": [["10", "cfg"], ["11", "cfg"]],
        "save": "19",
    },
    "cascade": {
        "workflow": "workflows/cascade.json",
        "positive": [["6", "text"]],
        # stage B and stage C share one negative encoder
        "negative": [["7", "text"]],
        "seed": [["3", "seed"], ["33", "seed"]],
        "batch": ["34", "batch_size"],
        # stage C only; stage B (node 33) keeps its own cfg
        "cfg": [["3", "cfg"]],
        "save": "9",
    },
}


def _cache_key(backend: str, prompt: str, seed: int, batch: int, workflow: dict,
               negative: str = DEFAULT_NEGATIVE, cfg: float | None = None) -> str:
    """``cfg=None`` means the workflow's own value; pass it already normalised (see ``ComfyGenerator._cfg``)."""
    wf_hash = hashlib.sha256(json.dumps(workflow, sort_keys=True).encode()).hexdigest()[:12]
    key = [backend, prompt, seed, batch, wf_hash]
    if negative != DEFAULT_NEGATIVE:  # the default keeps the keys of images cached before negatives existed
        key.append(negative)
    if cfg is not None:  # likewise, the workflow's cfg keeps the old keys
        key.append(["cfg", float(cfg)])
    payload = json.dumps(key, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass
class ComfyGenerator:
    backend: str
    workflow_path: Path
    cache_dir: Path
    url: str = "http://127.0.0.1:8188"
    batch_size: int = 4
    timeout: float = 1800.0
    poll_interval: float = 1.0
    node_map: dict = field(default_factory=dict)
    client_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        self.name = self.backend
        self.node_map = self.node_map or NODE_MAPS[self.backend]
        self.workflow = read_json(self.workflow_path)
        self.cache_dir = Path(self.cache_dir) / self.backend

    # -- HTTP helpers -------------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url.rstrip("/") + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
        return json.loads(body) if body else {}

    def _get(self, path: str) -> bytes:
        with urllib.request.urlopen(self.url.rstrip("/") + path, timeout=120) as r:
            return r.read()

    # -- workflow -----------------------------------------------------------------
    def _cfg(self, cfg: float | None) -> float | None:
        """None if ``cfg`` is unset or equals the workflow's value (so the cache key does not change)."""
        if cfg is None:
            return None
        nodes = self.node_map.get("cfg")
        if not nodes:
            raise ValueError(f"backend {self.backend!r} has no cfg nodes in its node map")
        if all(float(self.workflow[n]["inputs"][k]) == float(cfg) for n, k in nodes):
            return None
        return float(cfg)

    def build_workflow(self, prompt: str, seed: int, negative: str = DEFAULT_NEGATIVE,
                       cfg: float | None = None) -> dict:
        wf = copy.deepcopy(self.workflow)
        for node, key in self.node_map["positive"]:
            wf[node]["inputs"][key] = prompt
        for node, key in self.node_map.get("negative", []):
            wf[node]["inputs"][key] = negative
        for offset, (node, key) in enumerate(self.node_map["seed"]):
            wf[node]["inputs"][key] = (seed + offset) % 2**32
        if cfg is not None:
            for node, key in self.node_map["cfg"]:
                wf[node]["inputs"][key] = float(cfg)
        node, key = self.node_map["batch"]
        wf[node]["inputs"][key] = self.batch_size
        wf[self.node_map["save"]]["inputs"]["filename_prefix"] = f"vspr/{self.backend}"
        return wf

    def generate(self, prompt: str, seed: int, negative: str = DEFAULT_NEGATIVE,
                 cfg: float | None = None) -> list[str]:
        """``cfg=None`` keeps the workflow's guidance scale (SDXL 8, Cascade stage C 4)."""
        cfg = self._cfg(cfg)
        wf = self.build_workflow(prompt, seed, negative, cfg)
        key = _cache_key(self.backend, prompt, seed, self.batch_size, self.workflow, negative, cfg)
        out_dir = self.cache_dir / key
        meta_path = out_dir / "meta.json"
        if meta_path.exists():
            return [str(out_dir / f) for f in read_json(meta_path)["files"]]

        resp = self._post("/prompt", {"prompt": wf, "client_id": self.client_id})
        if "prompt_id" not in resp:
            raise RuntimeError(f"ComfyUI rejected the workflow: {resp}")
        prompt_id = resp["prompt_id"]

        deadline = time.time() + self.timeout
        while True:
            hist = json.loads(self._get(f"/history/{prompt_id}") or b"{}")
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI execution failed: {status.get('messages')}")
                if status.get("completed", True) and entry.get("outputs"):
                    break
            if time.time() > deadline:
                raise TimeoutError(f"ComfyUI did not finish prompt {prompt_id} in {self.timeout}s")
            time.sleep(self.poll_interval)

        images = entry["outputs"][self.node_map["save"]]["images"]
        out_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for i, img in enumerate(images):
            query = urllib.parse.urlencode(
                {"filename": img["filename"], "subfolder": img.get("subfolder", ""), "type": img.get("type", "output")}
            )
            name = f"{i}.png"
            (out_dir / name).write_bytes(self._get(f"/view?{query}"))
            files.append(name)
        write_json(meta_path, {"backend": self.backend, "prompt": prompt, "negative": negative, "cfg": cfg, "seed": seed,
                               "batch_size": self.batch_size, "files": files})
        return [str(out_dir / f) for f in files]

    def release(self) -> None:
        """Ask ComfyUI to unload its models so the (M)LLMs fit in GPU memory."""
        try:
            self._post("/free", {"unload_models": True, "free_memory": True})
        except Exception as err:  # noqa: BLE001 - best effort
            log.debug("ComfyUI /free failed: %s", err)
