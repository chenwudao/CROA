"""Model wrappers: surrogate encoders (CLIP x3 + DINOv2 + InternVL3-1B) and the
three black-box victim MLLMs.

All loads are explicit; nothing is imported at module import time except
torch/PIL. Victims are only ever called through `ask()` (image + text -> short
caption): the benchmark protocol is strictly black-box on the victim side.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config as C
from .common import unwrap_feats

IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)   # CLIP OpenAI stats
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


_LOCAL_CACHE = C.ROOT / "models_cache"

def _resolve_model_path(hub_id: str, local_dir_name: str):
    p = _LOCAL_CACHE / local_dir_name
    if (p / "config.json").exists():
        return str(p)
    return hub_id


class FeatureBank:
    """CLIP ensemble (B/16, B/32, Laion-G) + DINOv2-base image encoders."""

    def __init__(self, device: str = "cuda"):
        from transformers import CLIPModel, Dinov2Model, AutoTokenizer
        self.device = device
        self.clips = {}
        self.toks = {}
        local_names = {"B16": "clip-vit-base-patch16",
                       "B32": "clip-vit-base-patch32",
                       "Laion": "laion-clip-vit-g-14"}
        for key, mid in C.CLIP_MODELS.items():
            load_path = _resolve_model_path(mid, local_names.get(key, mid.replace("/", "-")))
            self.clips[key] = CLIPModel.from_pretrained(load_path, dtype=torch.float32).to(device).eval()
            self.toks[key] = AutoTokenizer.from_pretrained(load_path)
        dino_path = _resolve_model_path(C.DINO_MODEL, "dinov2-base")
        self.dino = Dinov2Model.from_pretrained(dino_path, dtype=torch.float32).to(device).eval()

    @torch.no_grad()
    def clip_embed(self, x224: torch.Tensor) -> dict[str, torch.Tensor]:
        """x224: (B,3,224,224) in [0,1]; returns L2-normalised image embeddings."""
        xn = (x224.to(self.device) - torch.tensor(IMAGENET_MEAN, device=x224.device)
              .view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD, device=x224.device).view(1, 3, 1, 1)
        out = {}
        for key, m in self.clips.items():
            emb = unwrap_feats(m.get_image_features(pixel_values=xn))
            out[key] = F.normalize(emb.float(), dim=-1)
        return out

    @torch.no_grad()
    def dino_embed(self, x224: torch.Tensor) -> torch.Tensor:
        xn = (x224.to(self.device) - torch.tensor(IMAGENET_MEAN, device=x224.device)
              .view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD, device=x224.device).view(1, 3, 1, 1)
        cls = self.dino(pixel_values=xn).last_hidden_state[:, 0]
        return F.normalize(cls.float(), dim=-1)


class InternVLSurrogate:
    """InternVL3-1B-hf: the third MPC paradigm branch (projected visual tokens)."""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.device = device
        ivl_path = _resolve_model_path(C.INTERNVL_SURROGATE, "InternVL3-1B-hf")
        self.processor = AutoProcessor.from_pretrained(ivl_path)
        self.model = (AutoModelForImageTextToText.from_pretrained(
            ivl_path, dtype=torch.float16, attn_implementation="eager",
        ).to(device).eval())


class QwenVictim:
    """Qwen2.5-VL-7B-Instruct victim (black-box chat protocol)."""

    def __init__(self, device: str = "cuda", model_id: str | None = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model_id = model_id or C.VICTIMS["qwen2.5vl7b"]
        self.model_id = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16,
        ).to(device).eval())

    @torch.no_grad()
    def ask(self, pil: Image.Image, prompt: str, max_new_tokens: int = 32) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=[pil.convert("RGB")],
                                return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(new, skip_special_tokens=True).strip()


class LlavaVictim:
    """LLaVA-1.5-7B victim (LlavaForConditionalGeneration)."""

    def __init__(self, device: str = "cuda", model_id: str | None = None):
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        model_id = model_id or C.VICTIMS["llava15-7b"]
        self.model_id = model_id
        self.device = device
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id, dtype=torch.bfloat16, local_files_only=True
            ).to(device).eval()
        except Exception:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id, dtype=torch.bfloat16
            ).to(device).eval()

    @torch.no_grad()
    def ask(self, pil: Image.Image, prompt: str, max_new_tokens: int = 32) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=[pil.convert("RGB")],
                                return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(new, skip_special_tokens=True).strip()


class InternVLVictim:
    """InternVL3-8B victim."""

    def __init__(self, device: str = "cuda", model_id: str | None = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model_id = model_id or C.VICTIMS["internvl3-8b"]
        self.model_id = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
        ).to(device).eval())

    @torch.no_grad()
    def ask(self, pil: Image.Image, prompt: str, max_new_tokens: int = 32) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=[pil.convert("RGB")],
                                return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(new, skip_special_tokens=True).strip()


class CommercialVictim:
    """Commercial closed-source MLLM victim (GPT-4o, GPT-4o-mini via API)."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.model_name = model_name
        settings = C.load_api_settings()
        if not settings["api_key"]:
            raise KeyError("OPENAI_API_KEY not found (env or configs/api_config.json)")
        kwargs = {"api_key": settings["api_key"], "timeout": 30.0}
        if settings["base_url"]:
            kwargs["base_url"] = settings["base_url"]
        self.client = OpenAI(**kwargs)

    def ask(self, pil: Image.Image, prompt: str, max_new_tokens: int = 300) -> str:
        import base64
        import io
        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }]
        # Reasoning models (like gpt-5) consume internal thinking tokens (often 500-1000) before text;
        # guarantee >=2048 tokens so completion is not truncated.
        actual_tokens = max(max_new_tokens, 2048)
        if any(r in self.model_name.lower() for r in ("gpt-5", "o1", "o3")):
            attempts = [
                {"max_completion_tokens": actual_tokens},          # gpt-5 / reasoning family
                {"max_tokens": actual_tokens},                     # fallback
            ]
        else:
            attempts = [
                {"max_tokens": actual_tokens},                     # claude / gemini / general via relay (instant)
                {"max_tokens": actual_tokens, "temperature": 0.0}, # fallback if supported
                {"max_completion_tokens": actual_tokens},
            ]
        last = None
        for extra in attempts:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, **extra)
                content = resp.choices[0].message.content
                if content and content.strip():
                    return content.strip()
            except Exception as e:
                last = e
                if "429" in str(e) or "rate_limit" in str(e):
                    raise          # rate limits bubble to the caller's retry loop
        if last is not None:
            raise last
        return ""


_VICTIM_LOCAL_NAMES = {
    "qwen2.5vl7b": "Qwen2.5-VL-7B-Instruct",
    "llava15-7b": "llava-1.5-7b-hf",
    "internvl3-8b": "InternVL3-8B-hf",
}


def make_victim(name: str, device: str = "cuda"):
    if name not in C.VICTIMS:
        raise ValueError(f"unknown victim '{name}'; choose from {list(C.VICTIMS)}")
    if name in ("gpt-5", "gpt-5.6-luna", "gemini-3.5-flash-lite", "claude-4.5-haiku", "claude-sonnet-4-5", "gpt-4o", "gpt-4o-mini"):
        return CommercialVictim(model_name=C.VICTIMS[name])

    local_name = _VICTIM_LOCAL_NAMES[name]
    model_id = _resolve_model_path(C.VICTIMS[name], local_name)
    if "llava" in name.lower():
        return LlavaVictim(device, model_id=model_id)
    if "internvl" in name.lower():
        return InternVLVictim(device, model_id=model_id)
    return QwenVictim(device, model_id=model_id)


# ==============================================================================
# AnyAttack Official (CVPR 2025) Decoder Architecture
# ==============================================================================
class AnyAttackEfficientAttention(torch.nn.Module):
    def __init__(self, in_channels: int, key_channels: int, head_count: int, value_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = torch.nn.Conv2d(in_channels, key_channels, 1)
        self.queries = torch.nn.Conv2d(in_channels, key_channels, 1)
        self.values = torch.nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = torch.nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        return reprojected_value + input_


class AnyAttackResBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, key_channels: int, head_count: int, value_channels: int):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.LeakyReLU(0.2, inplace=True)
        self.attention = AnyAttackEfficientAttention(out_channels, key_channels, head_count, value_channels)
        if in_channels != out_channels:
            self.skip_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip_conv = torch.nn.Identity()

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        residual = self.skip_conv(residual)
        out += residual
        return self.activation(out)


class AnyAttackUpBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.up(x)
        return self.activation(self.bn(self.conv(x)))


class AnyAttackDecoder(torch.nn.Module):
    def __init__(self, embed_dim: int = 512, img_channels: int = 3, img_size: int = 224):
        super().__init__()
        self.embedding_dim = embed_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.init_size = img_size // 16

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 256 * self.init_size ** 2)
        )

        self.upsample_blocks = torch.nn.ModuleList([
            AnyAttackResBlock(256, 256, 64, 8, 256),
            AnyAttackUpBlock(256, 128),
            AnyAttackResBlock(128, 128, 32, 8, 128),
            AnyAttackUpBlock(128, 64),
            AnyAttackResBlock(64, 64, 16, 8, 64),
            AnyAttackUpBlock(64, 32),
            AnyAttackResBlock(32, 32, 8, 8, 32),
            AnyAttackUpBlock(32, 16),
            AnyAttackResBlock(16, 16, 4, 8, 16)
        ])

        self.final_conv = torch.nn.Conv2d(16, img_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = torch.nn.Tanh()

    def forward(self, embedding):
        embedding = embedding.view(embedding.size(0), -1)
        out = self.fc(embedding.float())
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        for block in self.upsample_blocks:
            out = block(out)
        return self.final_conv(out)


def load_anyattack_decoder(device: str = "cuda", variant: str = "cos"):
    dec = AnyAttackDecoder(embed_dim=512).to(device).eval()
    ckpt_name = "flickr30k_cos.pt" if variant == "cos" else "flickr30k_bi.pt"
    ckpt_path = _LOCAL_CACHE / "anyattack" / ckpt_name
    if not ckpt_path.exists():
        ckpt_path = C.ROOT / "models_cache" / "anyattack" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"AnyAttack weights not found at {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("decoder_state_dict", state)
    dec.load_state_dict(sd, strict=True)
    return dec


