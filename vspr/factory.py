"""Build pipeline components from a config dictionary."""

from __future__ import annotations

from pathlib import Path

from .comfy import NODE_MAPS, ComfyGenerator
from .config import resolve
from .functions import RoleConfig
from .judge import Judge
from .io_utils import load_descriptions, slugify
from .llm import OllamaLLM
from .pipeline import VSPRPipeline, VSPRSettings


def run_name(cfg: dict) -> str:
    if cfg.get("run_name"):
        return cfg["run_name"]
    name = f"{cfg['t2i']['backend']}_{slugify(cfg['llm']['reasoner'])}"
    if not cfg["vspr"].get("use_vqa", True):
        name += "_novqa"
    return name


def run_dir(cfg: dict) -> Path:
    return resolve(cfg["output_root"]) / "vspr" / run_name(cfg)


def descriptions(cfg: dict) -> list[str]:
    descs = load_descriptions(resolve(cfg["dataset"]))
    return descs[: cfg["limit"]] if cfg.get("limit") else descs


def build_generator(cfg: dict) -> ComfyGenerator:
    t2i = cfg["t2i"]
    backend = t2i["backend"]
    return ComfyGenerator(
        backend=backend,
        workflow_path=resolve(t2i.get("workflow") or NODE_MAPS[backend]["workflow"]),
        cache_dir=resolve(t2i["cache_dir"]),
        url=t2i["comfy_url"],
        batch_size=int(t2i["batch_size"]),
        timeout=float(t2i["timeout"]),
    )


def build_llm(cfg: dict) -> OllamaLLM:
    llm = cfg["llm"]
    return OllamaLLM(host=llm["host"], retries=int(llm["retries"]), think=llm.get("think", "auto"))


def build_reasoner(cfg: dict) -> RoleConfig:
    llm = cfg["llm"]
    return RoleConfig(llm["reasoner"], dict(llm.get("options", {}).get("reasoner", {})))


def build_judge(cfg: dict) -> Judge:
    """phi_que / phi_ans: fixed questions on every image (decoding: temperature 0, seed 0)."""
    return Judge(cfg["llm"]["mllm"], cfg["llm"]["host"])


def build_settings(cfg: dict) -> VSPRSettings:
    v = cfg["vspr"]
    return VSPRSettings(
        max_iterations=int(v["max_iterations"]),
        batch_size=int(cfg["t2i"]["batch_size"]),
        use_vqa=bool(v["use_vqa"]),
        dms_weight=float(v["dms_weight"]),
        base_seed=int(cfg["t2i"]["base_seed"]),
        negative=str(cfg["t2i"].get("negative", "text, watermark")),
    )


def build_pipeline(cfg: dict) -> VSPRPipeline:
    settings = build_settings(cfg)
    judge = build_judge(cfg) if settings.use_vqa else None
    return VSPRPipeline(build_llm(cfg), build_generator(cfg), build_reasoner(cfg), judge,
                        settings, run_dir(cfg))
