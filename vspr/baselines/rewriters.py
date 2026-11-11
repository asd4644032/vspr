"""Single-shot prompt rewriters used as baselines.

none       No alignment: the description D is used as the prompt.
gpt4       GPT-4o ("gpt_4o_new" in the old experiments): gpt-4o with the system prompt of
           the original gpt_prompt_generation.py (gpt_system_prompt.txt)
           (API key from the OPENAI_API_KEY environment variable; never hard-code it).
beautiful  BeautifulPrompt (alibaba-pai/pai-bloom-1b1-text2prompt-sd).
magic      MagicPrompt (Gustavosta/MagicPrompt-Stable-Diffusion).
promptist  Promptist (microsoft/Promptist).
external   Prompts produced by another code base (e.g. SSPO, PAG), read from a file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from ..io_utils import read_json, stable_seed

GPT_SYSTEM_PROMPT = (Path(__file__).parent / "gpt_system_prompt.txt").read_text(encoding="utf-8")


class Rewriter:
    name = "base"

    def __call__(self, description: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        """Free GPU memory before the diffusion model runs."""


class NoAlignment(Rewriter):
    name = "none"

    def __call__(self, description: str) -> str:
        return description


class OpenAIRewriter(Rewriter):
    name = "gpt4"

    def __init__(self, model: str = "gpt-4o", seed: int = 42) -> None:
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set the OPENAI_API_KEY environment variable for the gpt4 baseline.")
        self.client = OpenAI()
        self.model = model
        self.seed = seed

    def __call__(self, description: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": GPT_SYSTEM_PROMPT},
                      {"role": "user", "content": f"Description : {description}"}],
            temperature=1.0,
            top_p=1.0,
            max_tokens=2048,
            seed=stable_seed(self.seed, description),
        )
        return " ".join(resp.choices[0].message.content.split())


class _HFCausalRewriter(Rewriter):
    model_id = ""

    def __init__(self, seed: int = 42, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).eval().to(self.device)
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.seed = seed

    def _generate(self, text: str, description: str, **gen) -> str:
        self.torch.manual_seed(stable_seed(self.seed, description))
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(ids, pad_token_id=self.tok.pad_token_id, **gen)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class BeautifulPrompt(_HFCausalRewriter):
    name = "beautiful"
    model_id = "alibaba-pai/pai-bloom-1b1-text2prompt-sd"

    def __call__(self, description: str) -> str:
        text = ("Instruction: Give a simple description of the image to generate a drawing prompt.\n"
                f"Input: {description}\nOutput:")
        out = self._generate(text, description, max_new_tokens=128, do_sample=True, temperature=1.0,
                             top_k=50, top_p=0.95, repetition_penalty=1.2)
        return " ".join(out.split()) or description


class MagicPrompt(_HFCausalRewriter):
    name = "magic"
    model_id = "Gustavosta/MagicPrompt-Stable-Diffusion"

    def __call__(self, description: str) -> str:
        # MagicPrompt continues the input text, so the result starts with D.
        out = self._generate(description, description, max_length=90, min_length=20, do_sample=True,
                             temperature=1.0, top_k=12, repetition_penalty=1.0)
        return " ".join((description + out).split())


class Promptist(_HFCausalRewriter):
    name = "promptist"
    model_id = "microsoft/Promptist"

    def __init__(self, seed: int = 42, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).eval().to(self.device)
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.seed = seed

    def __call__(self, description: str) -> str:
        # Decoding settings of the official Promptist demo.
        out = self._generate(description.strip() + " Rephrase:", description, num_beams=8,
                             num_return_sequences=1, max_new_tokens=75,
                             eos_token_id=self.tok.eos_token_id, length_penalty=-1.0)
        return " ".join(out.split()) or description


class ExternalPrompts(Rewriter):
    """Prompts from another implementation.

    Accepted formats: a JSON object {description: prompt}, a JSON list of
    {"description", "prompt"}, a JSONL file of such records, or a directory of
    ``<description>.json`` files containing a "prompt" key (layout of the old repo)
    or ``<description>.txt`` files containing the prompt (output of the PAG code).
    The PAG and SSPO prompts used in the paper are in data/baseline_prompts/.
    """

    name = "external"

    def __init__(self, path: str | os.PathLike, name: str = "external") -> None:
        self.name = name
        self.table = self._load(Path(path))

    @staticmethod
    def _load(path: Path) -> dict[str, str]:
        if path.is_dir():
            table = {p.stem: read_json(p)["prompt"] for p in path.glob("*.json")}
            for p in path.glob("*.txt"):
                table.setdefault(p.stem, " ".join(p.read_text(encoding="utf-8").split()))
            return table
        if path.suffix == ".jsonl":
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            return {r["description"]: r["prompt"] for r in rows}
        data = read_json(path)
        if isinstance(data, dict):
            return dict(data)
        return {r["description"]: r["prompt"] for r in data}

    def __call__(self, description: str) -> str:
        if description not in self.table:
            raise KeyError(f"No external prompt for {description!r}")
        return self.table[description]


BASELINES: dict[str, Callable[..., Rewriter]] = {
    "none": NoAlignment,
    "gpt4": OpenAIRewriter,
    "beautiful": BeautifulPrompt,
    "magic": MagicPrompt,
    "promptist": Promptist,
}


def build_rewriter(spec: str, cfg: dict) -> Rewriter:
    """``spec`` is a baseline name or ``name=path`` for external prompts (e.g. ``sspo=prompts.json``)."""
    if "=" in spec:
        name, path = spec.split("=", 1)
        return ExternalPrompts(path, name=name)
    seed = int(cfg["t2i"]["base_seed"])
    if spec == "none":
        return NoAlignment()
    if spec == "gpt4":
        return OpenAIRewriter(model=cfg["baselines"]["openai_model"], seed=seed)
    return BASELINES[spec](seed=seed)
