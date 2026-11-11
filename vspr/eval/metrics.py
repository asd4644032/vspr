"""Per-image metrics. All scorers take (PIL image, caption) pairs and return raw values;
scaling for the tables is applied in scripts/evaluate.py (see vspr.eval.SCALE).

The caption C is always the user description D, never the engineered prompt.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
from PIL import Image

AESTHETIC_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/"
                 "sac%2Blogos%2Bava1-l14-linearMSE.pth")
AESTHETIC_V1_URL = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def clip_image_embeds(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Projected CLIP image embedding, computed from the sub-modules: the CLIPModel feature
    helpers return a tensor in transformers 4.x but a ModelOutput in 5.x."""
    pooled = model.vision_model(pixel_values=pixel_values).pooler_output
    return model.visual_projection(pooled)


def clip_text_embeds(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    pooled = model.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
    return model.text_projection(pooled)


class CLIPScore:
    """CLIPScore(I, C) = max(100 * cos(E_I, E_C), 0) with CLIP ViT-B/16 (Eq. 7)."""

    key = "CS"

    def __init__(self, model_id: str = "openai/clip-vit-base-patch16", device: str | None = None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or _device()
        self.model = CLIPModel.from_pretrained(model_id).eval().to(self.device)
        self.proc = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], captions: list[str]) -> list[float]:
        inp = self.proc(text=captions, images=images, return_tensors="pt", padding=True, truncation=True)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        img = torch.nn.functional.normalize(clip_image_embeds(self.model, inp["pixel_values"]), dim=-1)
        txt = torch.nn.functional.normalize(
            clip_text_embeds(self.model, inp["input_ids"], inp["attention_mask"]), dim=-1)
        return (100.0 * (img * txt).sum(-1)).clamp(min=0).tolist()


class BLIP2ITCITM:
    """BLIP-2 image-text contrastive similarity (ITC) and matching probability (ITM, Eq. 8)."""

    keys = ("ITC", "ITM")

    def __init__(self, model_id: str = "Salesforce/blip2-itm-vit-g", device: str | None = None):
        try:
            from transformers import AutoProcessor, Blip2ForImageTextRetrieval
        except ImportError as err:  # added in transformers 4.46
            import transformers

            raise ImportError(
                f"BLIP-2 ITC/ITM needs transformers>=4.48 (this environment has {transformers.__version__}). "
                "Run: pip install -U \"transformers>=4.48\" \"accelerate>=0.30\" "
                "(or use a separate environment from the UniDet one)") from err

        self.device = device or _device()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = Blip2ForImageTextRetrieval.from_pretrained(model_id, torch_dtype=dtype).eval().to(self.device)
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, images, captions) -> dict[str, list[float]]:
        inp = self.proc(images=images, text=captions, return_tensors="pt", padding=True)
        inp = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
               for k, v in inp.items()}
        itm = self.model(**inp, use_image_text_matching_head=True).logits_per_image
        p_match = torch.softmax(itm.float(), dim=-1)[:, 1]
        out = self.model(**inp, use_image_text_matching_head=False)
        # logits_per_image is (images x texts); paired similarity is the diagonal.
        itc = out.logits_per_image.float().diagonal()
        return {"ITC": itc.tolist(), "ITM": p_match.tolist()}


class BLIPVQAScore:
    """BLIP-VQA(I, C) = p(yes | I, q_C) - p(no | I, q_C) (Eq. 9).

    Same definition as the original evaluate.py (LAVIS ``blip_vqa``/``vqav2`` with a
    custom ``_rank_answers``): p(.) is the probability of the answer's first token
    under the answer decoder, softmax over the whole vocabulary (not renormalized
    over the two candidates), and q_C uses the same question text.
    """

    key = "VQA"
    QUESTION = "Does this figure show ‘{c}’? Please answer yes or no."

    def __init__(self, model_id: str = "Salesforce/blip-vqa-base", device: str | None = None):
        from transformers import BlipForQuestionAnswering, BlipProcessor

        self.device = device or _device()
        self.model = BlipForQuestionAnswering.from_pretrained(model_id).eval().to(self.device)
        self.proc = BlipProcessor.from_pretrained(model_id)
        tok = self.proc.tokenizer
        self.bos = self.model.config.text_config.bos_token_id
        self.yes_id = tok("yes", add_special_tokens=False).input_ids[0]
        self.no_id = tok("no", add_special_tokens=False).input_ids[0]

    @torch.no_grad()
    def __call__(self, images, captions) -> list[float]:
        qs = [self.QUESTION.format(c=c.lower()) for c in captions]
        inp = self.proc(images=images, text=qs, return_tensors="pt", padding=True).to(self.device)
        vis = self.model.vision_model(pixel_values=inp["pixel_values"])[0]
        vis_mask = torch.ones(vis.shape[:-1], dtype=torch.long, device=self.device)
        q = self.model.text_encoder(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                                    encoder_hidden_states=vis, encoder_attention_mask=vis_mask)[0]
        start = torch.full((q.shape[0], 1), self.bos, dtype=torch.long, device=self.device)
        logits = self.model.text_decoder(input_ids=start, encoder_hidden_states=q,
                                         encoder_attention_mask=inp["attention_mask"], return_dict=True).logits
        probs = torch.softmax(logits[:, 0].float(), dim=-1)
        return (probs[:, self.yes_id] - probs[:, self.no_id]).tolist()


class AestheticScore:
    """LAION aesthetic predictor on the normalized CLIP ViT-L/14 image embedding, range ~1-10.

    ``version="v1"`` (default): the linear head released with LAION-5B
    (sa_0_4_vit_l_14_linear.pth), as in the original evaluate.py and the paper's citation.
    ``version="v2"``: the "improved aesthetic predictor" MLP (sac+logos+ava1-l14-linearMSE.pth).
    """

    key = "AS"
    URLS = {"v1": AESTHETIC_V1_URL, "v2": AESTHETIC_URL}
    FILES = {"v1": "sa_0_4_vit_l_14_linear.pth", "v2": "sac+logos+ava1-l14-linearMSE.pth"}

    def __init__(self, version: str = "v1", weights: str | Path | None = None, device: str | None = None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or _device()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(self.device)
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        weights = Path(weights or Path.home() / ".cache" / "vspr" / self.FILES[version])
        if not weights.exists():
            weights.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.URLS[version], weights)
        state = torch.load(weights, map_location="cpu")
        if version == "v1":
            self.mlp = torch.nn.Linear(768, 1)
            self.mlp.load_state_dict(state)
        else:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2),
                torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
                torch.nn.Linear(64, 16), torch.nn.Linear(16, 1),
            )
            self.mlp.load_state_dict({k.replace("layers.", ""): v for k, v in state.items()})
        self.mlp.eval().to(self.device)

    @torch.no_grad()
    def __call__(self, images, captions=None) -> list[float]:
        pix = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        emb = torch.nn.functional.normalize(clip_image_embeds(self.clip, pix).float(), dim=-1)
        return self.mlp(emb).squeeze(-1).tolist()
