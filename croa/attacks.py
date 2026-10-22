"""The four benchmark arms. Three strictly-official baselines + FOA-Attack v2.

Verification notes (checked against the authors' released code and default
configs on 2026-09-10):

M-Attack (NeurIPS 2025, github.com/VILA-Lab/M-Attack, config/ensemble_3models.yaml):
  - Ensemble of the PROJECTED CLS embeddings (get_image_features, L2-normed)
    of CLIP ViT-B/16 + B/32 + Laion-G; loss = per-model cosine between the
    adversarial crop and the target crop, averaged over the 3 extractors.
  - "Local-to-local" = RandomResizedCrop(224, scale=[0.5, 0.9], ratio=[3/4, 4/3])
    applied to the adversarial image every step; target crops re-sampled from the
    same stream.
  - Plain FGSM (no momentum): delta += 1.0 * sign(grad), clamp [-16, 16], 300 steps.

FOA-Attack (NeurIPS 2025, github.com/jiaxiaojunQAQ/FOA-Attack, same config +
  surrogates/FeatureExtractors/Base.py, cluster_number=10):
  - Per model: global = CLS last_hidden_state (768d, L2-normed, NOT projected);
    local = kmeans(k=10, euclidean, iter_limit=100) centroids of the 196 patch
    tokens (768d, per-token L2-normed).
  - Per-model loss = Sinkhorn-OT(global_tgt, global_adv) + 0.2 * Sinkhorn-OT(
    centers_tgt, centers_adv); OT kernel exp(-(1-cos)/0.1), uniform marginals,
    <=100 iterations, tol 1e-2, cost = sum(T * sim).
  - Dynamic ensemble weighting: ratio_i = L_i / L_i_prev, w = K * softmax(
    ratios, T=1); previous losses lazily initialised with step-1 values.
  - Target crops RE-SAMPLED every iteration; FGSM 1.0 / 16 / 300 steps.

SSA (ECCV 2022 "Frequency Domain Model Augmentation for Adversarial Attack",
  github.com/yuyang-long/SSA, attack.py::Spectrum_Simulation_Attack; verified
  2026-09-17):
  - Per step, N=20 EOT samples of the adv image; each = gaussian noise
    sigma=16/255 + per-coefficient DCT-II amplitude mask U(1-rho,1+rho),
    rho=0.5, then inverse DCT. Gradient averaged over N; plain FGSM sign step
    (MI/TI/DI are commented out in the official script). Targeted adaptation:
    the classification CE is swapped for the benchmark's shared 3-CLIP-ensemble
  cosine objective; official 10-step eps/10 budget remapped to the unified
    300-step / alpha=1 schedule like every other arm.
  NOTE: SSA is ECCV 2022 (Long et al.) - the earlier "SSA-CWA (CVPR 2024)"
  attribution in this repo was wrong and has been corrected.

MPCAttack (CVPR 2026, github.com/LiYuanBoJNU/MPCAttack):
  - Three paradigms concatenated after global pooling: CLIP x3 (projected CLS),
    InternVL3-1B (256 projected tokens -> mean), DINOv2-base (CLS); all
    L2-normalised, fused by concat-then-cosine; one contrastive InfoNCE loss
    (lambda=0.6 text fusion, tau=0.2, omega=2); 0-255 domain, delta init
    (16/255)*randn, raw unnormalised momentum MI-FGSM, aspect-preserving crop
    every step (crop 0.5-0.9).
  - Default here runs the official target-side text fusion (0.6*img + 0.4*txt
    with the human-verified target caption y*), matching the paper's Flickr
    pipeline; pass mpc_text_fusion=False for the pure image-to-image variant.

FOA-Attack v2 (OURS, this repo):
  Mechanistic Innovations:
  1. Contrastive Source Manifold Inactivation (CSMI): Active source repulsion
     (-beta * Sim(adv, src), beta=0.5) to dismantle clean source semantics and
     eliminate the "double exposure" / source-retention failure mode seen on commercial MLLMs.
  2. Dual-Paradigm Decoupled Alignment (DPDA): Decouples the surrogate ensemble
     into Semantic Paradigm (3 CLIPs) and Geometric Paradigm (DINOv2) with balanced
     orthogonal weighting (0.5 CLIP + 0.5 DINO) and enriched local patch topology (gamma=0.4),
     preventing collinear CLIP majority voting from suppressing DINOv2.
  3. Momentum Iterative Optimal Transport (MI-OT-FGSM): Normalized gradient momentum
     accumulation (mu=0.9) to stabilize stochastic variance from random crops and
     k-means clustering, generating smooth, low-frequency perturbations robust to
     MLLM visual resizers and input guards.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

from . import config as C
from .common import (_crop_official, _kmeans, _norm255, _rrc_crop,
                     _sinkhorn_ot, pil_to255)


# ==============================================================================
# Baseline 1: MPCAttack (official faithful port)
# ==============================================================================
class MPCOfficial:
    """5-branch concat + InfoNCE ascent, official MI-FGSM schedule."""

    BRANCHES = ("clip_B16", "clip_B32", "clip_Laion", "internvl", "dino")

    def __init__(self, bank, ivl, device: str = "cuda"):
        self.bank = bank
        self.ivl = ivl
        self.device = device

    # ----- branch features -----
    def clip_parts(self, x255: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for key, m in self.bank.clips.items():
            f = m.get_image_features(pixel_values=_norm255(x255, C.ATTACK_RES))
            out[f"clip_{key}"] = F.normalize(f.float(), dim=-1)
        return out

    def internvl_part(self, x255: torch.Tensor) -> torch.Tensor:
        px = F.interpolate(x255, size=(C.ATTACK_RES, C.ATTACK_RES), mode="bicubic", align_corners=False)
        px = F.interpolate(px, size=(C.INTERNVL_RES, C.INTERNVL_RES), mode="bilinear", align_corners=False)
        px = torch.clamp(px, 0.0, 255.0) / 255.0
        m = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=px.device).view(1, 3, 1, 1)
        s = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=px.device).view(1, 3, 1, 1)
        px = ((px - m) / s).to(torch.bfloat16)
        feats = self.ivl.model.get_image_features(pixel_values=px)  # [1,256,896]
        feats = feats / feats.norm(dim=1, keepdim=True)
        return feats.mean(dim=1).float()

    def dino_part(self, x255: torch.Tensor) -> torch.Tensor:
        cls = self.bank.dino(pixel_values=_norm255(x255, C.ATTACK_RES)).last_hidden_state[:, 0]
        return F.normalize(cls.float(), dim=-1)

    def branch_parts(self, x255: torch.Tensor) -> dict[str, torch.Tensor]:
        parts = self.clip_parts(x255)
        parts["internvl"] = self.internvl_part(x255)
        parts["dino"] = self.dino_part(x255)
        return {b: parts[b] for b in self.BRANCHES}

    # ----- references (official 0.6 img + 0.4 txt fusion when text is given) -----
    @torch.no_grad()
    def reference(self, pil: Image.Image, text: str = "") -> dict[str, torch.Tensor]:
        x255 = pil_to255(pil).to(self.device)
        parts = self.branch_parts(x255)
        if text:
            for key, m in self.bank.clips.items():
                tok = self.bank.toks[key]([text], padding=True, truncation=True,
                                          return_tensors="pt").to(self.device)
                t = m.get_text_features(**tok)
                txt = F.normalize(t.float(), dim=-1)
                parts[f"clip_{key}"] = C.LAMBDA_TXT * parts[f"clip_{key}"] + (1 - C.LAMBDA_TXT) * txt
        return parts

    def score(self, a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> torch.Tensor:
        za = F.normalize(torch.cat([a[k] for k in self.BRANCHES], dim=-1), dim=-1)
        zb = F.normalize(torch.cat([b[k] for k in self.BRANCHES], dim=-1), dim=-1)
        return torch.sum(za * zb, dim=-1)

    def attack(self, source_pil: Image.Image, tgt_pil: Image.Image,
               steps: int, seed: int, y_star: str = "",
               mpc_text_fusion: bool = True) -> dict:
        tgt_ref = self.reference(tgt_pil, text=(y_star if mpc_text_fusion else ""))
        src_ref = self.reference(source_pil)

        x0 = pil_to255(source_pil).to(self.device)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        delta = (C.EPSILON * torch.randn(x0.shape, generator=gen)).to(self.device)
        momentum = torch.zeros_like(x0)
        losses = []
        s_pos = torch.zeros(1, device=self.device)
        s_neg = torch.zeros(1, device=self.device)
        for _ in range(steps):
            delta.requires_grad_(True)
            adv = x0 + delta
            cropped = _crop_official(adv, C.CROP_SCALE, gen)
            parts = self.branch_parts(cropped)
            s_pos = self.score(parts, tgt_ref)
            s_neg = self.score(parts, src_ref)
            logits = torch.stack([s_pos * C.OMEGA / C.TAU, s_neg / C.TAU], dim=1)
            loss = -F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=x0.device))
            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
            with torch.no_grad():                    # official: raw unnormalised momentum
                momentum = C.MOMENTUM * momentum + grad
                delta = torch.clamp(delta + 1.0 * momentum.sign(), -C.EPSILON * 255, C.EPSILON * 255)
            losses.append(float(loss.detach()))
        adv = torch.clamp((x0 + delta.detach()) / 255.0, 0.0, 1.0)
        return {"adv": adv.detach()[0], "losses": losses,
                "linf": float(delta.detach().abs().max()) / 255.0,
                "final_s_pos": float(s_pos.detach()), "final_s_neg": float(s_neg.detach())}


# ==============================================================================
# Baselines 2/3 + Ours: M-Attack official, FOA-Attack official, FOA-Attack v2
# ==============================================================================
class CLIPFamilyAttacks:
    """CLIP-ensemble official attacks; reuses the shared FeatureBank (B16/B32/Laion
    + DINOv2 for the v2 branch)."""

    def __init__(self, bank, device: str = "cuda"):
        self.bank = bank
        self.device = device
        self.keys = ["B16", "B32", "Laion"]
        self.keys_v2 = ["B16", "B32", "Laion", "Dino"]

    # ----- feature paths (faithful to each repo's extractor) -----
    def _global_projected(self, x255: torch.Tensor) -> dict:
        """M-Attack path: get_image_features (projected CLS, L2-normed)."""
        px = _norm255(x255, C.ATTACK_RES)
        return {k: F.normalize(self.bank.clips[k].get_image_features(pixel_values=px).float(), dim=-1)
                for k in self.keys}

    def _global_local_hidden(self, x255: torch.Tensor, keys: list[str]) -> dict:
        """FOA path: CLS last_hidden_state (768d) + per-token-normed patch tokens.
        The DINO branch (v2) exposes the identical CLS+patches interface."""
        px = _norm255(x255, C.ATTACK_RES)
        out = {}
        for k in keys:
            if k == "Dino":
                h = self.bank.dino(pixel_values=px).last_hidden_state.float()
            else:
                h = self.bank.clips[k].vision_model(pixel_values=px).last_hidden_state.float()
            out[k] = (F.normalize(h[:, 0], dim=-1), F.normalize(h[:, 1:], dim=-1))
        return out

    # ----- M-Attack official (targeted, FGSM) -----
    def attack_mattack_official(self, source_pil: Image.Image, target_pil: Image.Image,
                                steps: int = C.OFFICIAL_STEPS, seed: int = 20260827,
                                alpha: float = 1.0, epsilon: float = 16.0) -> dict:
        x0 = pil_to255(source_pil).to(self.device)
        xt = pil_to255(target_pil).to(self.device)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        delta = torch.zeros_like(x0)                    # FGSM convention, zero init
        losses = []

        for _ in range(steps):
            delta.requires_grad_(True)
            adv_crop = _rrc_crop(x0 + delta, gen)
            # Re-crop target every step: strictly matches author's set_ground_truth(target_crop(image_tgt))
            with torch.no_grad():
                tgt_crop = _rrc_crop(xt, gen)
                gt = self._global_projected(tgt_crop)
            feats = self._global_projected(adv_crop)
            loss = sum(torch.sum(feats[k] * gt[k], dim=-1).mean() for k in self.keys) / len(self.keys)
            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
            with torch.no_grad():                       # plain FGSM: no momentum
                delta = torch.clamp(delta + alpha * grad.sign(), -epsilon, epsilon)
            losses.append(float(loss.detach()))

        adv = torch.clamp((x0 + delta.detach()) / 255.0, 0.0, 1.0)
        return {"adv": adv.detach()[0], "losses": losses,
                "linf": float(delta.detach().abs().max()) / 255.0,
                "final_s_pos": float(losses[-1]), "final_s_neg": 0.0}

    # ----- FOA core loop (official; keys decide official vs v2) -----
    def _attack_foa(self, source_pil: Image.Image, target_pil: Image.Image,
                    keys: list[str], steps: int, seed: int,
                    alpha: float = 1.0, epsilon: float = 16.0,
                    cluster_number: int = C.FOA_CLUSTER_NUMBER) -> dict:
        x0 = pil_to255(source_pil).to(self.device)
        xt = pil_to255(target_pil).to(self.device)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        gen_km = torch.Generator(device="cpu").manual_seed(seed + 1)
        delta = torch.zeros_like(x0)
        losses = []
        prev_losses: list[torch.Tensor] | None = None

        def kmeans_tokens(tokens: torch.Tensor) -> torch.Tensor:
            return _kmeans(tokens, cluster_number, gen_km)

        for _ in range(steps):
            delta.requires_grad_(True)
            adv_crop = _rrc_crop(x0 + delta, gen)
            tgt_crop = _rrc_crop(xt, gen)               # target re-cropped EVERY iteration
            adv_gl = self._global_local_hidden(adv_crop, keys)
            with torch.no_grad():
                tgt_gl = self._global_local_hidden(tgt_crop, keys)

            per_model = []
            for k in keys:
                a_g, a_loc = adv_gl[k]
                t_g, t_loc = tgt_gl[k]
                g_loss = _sinkhorn_ot(a_g, t_g.squeeze(0),
                                      eps=C.FOA_SINKHORN_EPS, iters=C.FOA_SINKHORN_ITERS,
                                      tol=C.FOA_SINKHORN_TOL)
                l_loss = _sinkhorn_ot(kmeans_tokens(a_loc), kmeans_tokens(t_loc),
                                      eps=C.FOA_SINKHORN_EPS, iters=C.FOA_SINKHORN_ITERS,
                                      tol=C.FOA_SINKHORN_TOL)
                per_model.append(g_loss + C.FOA_LOCAL_WEIGHT * l_loss)

            if prev_losses is None:                     # lazy init on step 1
                prev_losses = [l.detach() for l in per_model]
            ratios = torch.tensor([l.item() / (p.item() + 1e-8)
                                   for l, p in zip(per_model, prev_losses)])
            w = torch.softmax(ratios, dim=0) * len(keys)
            loss = sum((float(w[i]) * per_model[i] for i in range(len(keys))))
            prev_losses = [l.detach() for l in per_model]

            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
            with torch.no_grad():                       # plain FGSM: no momentum
                delta = torch.clamp(delta + alpha * grad.sign(), -epsilon, epsilon)
            losses.append(float(loss.detach()))

        adv = torch.clamp((x0 + delta.detach()) / 255.0, 0.0, 1.0)
        return {"adv": adv.detach()[0], "losses": losses,
                "linf": float(delta.detach().abs().max()) / 255.0,
                "final_s_pos": float(losses[-1]), "final_s_neg": 0.0}

    def attack_foa_official(self, source_pil: Image.Image, target_pil: Image.Image,
                            steps: int = C.OFFICIAL_STEPS, seed: int = 20260827) -> dict:
        """Strictly official FOA-Attack: 3 CLIP extractors."""
        return self._attack_foa(source_pil, target_pil, self.keys, steps, seed)

    def attack_foa_v2(self, source_pil: Image.Image, target_pil: Image.Image,
                      steps: int = C.OFFICIAL_STEPS, seed: int = 20260827,
                      alpha: float = 1.0, epsilon: float = 16.0,
                      beta_rep: float = 0.5, momentum_decay: float = 0.9,
                      cluster_number: int = C.FOA_CLUSTER_NUMBER,
                      dino_local_weight: float = 0.4,
                      clip_local_weight: float = 0.2) -> dict:
        """FOA-Attack v2 (Ours): Dual-Pass Multi-Scale Contrastive Optimal Transport."""
        x0 = pil_to255(source_pil).to(self.device)
        xt = pil_to255(target_pil).to(self.device)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        gen_km = torch.Generator(device="cpu").manual_seed(seed + 1)
        delta = torch.zeros_like(x0)
        momentum = torch.zeros_like(x0)
        losses = []

        def kmeans_tokens(tokens: torch.Tensor) -> torch.Tensor:
            return _kmeans(tokens, cluster_number, gen_km)

        with torch.no_grad():
            s_px = _norm255(x0, C.ATTACK_RES)
            src_gl = {}
            for k in self.keys_v2:
                m = self.bank.dino if k == "Dino" else self.bank.clips[k].vision_model
                h = m(pixel_values=s_px).last_hidden_state.float()
                src_gl[k] = (F.normalize(h[:, 0], dim=-1), F.normalize(h[:, 1:], dim=-1))

        d_rep_loss = torch.tensor(0.0, device=self.device)
        for _ in range(steps):
            delta.requires_grad_(True)
            c1 = _norm255(_rrc_crop(x0 + delta, gen), C.ATTACK_RES)
            c2 = _norm255(_rrc_crop(x0 + delta, gen), C.ATTACK_RES)
            adv_px = torch.cat([c1, c2], dim=0)

            tc1 = _norm255(_rrc_crop(xt, gen), C.ATTACK_RES)
            tc2 = _norm255(_rrc_crop(xt, gen), C.ATTACK_RES)
            tgt_px = torch.cat([tc1, tc2], dim=0)

            adv_gl = {}
            tgt_gl = {}
            for k in self.keys_v2:
                m = self.bank.dino if k == "Dino" else self.bank.clips[k].vision_model
                h_a = m(pixel_values=adv_px).last_hidden_state.float()
                adv_gl[k] = (F.normalize(h_a[:, 0], dim=-1), F.normalize(h_a[:, 1:], dim=-1))
                with torch.no_grad():
                    h_t = m(pixel_values=tgt_px).last_hidden_state.float()
                    tgt_gl[k] = (F.normalize(h_t[:, 0], dim=-1), F.normalize(h_t[:, 1:], dim=-1))

            crop_losses = []
            for i in range(2):
                # Branch 1: CLIP Ensemble
                clip_tgt_losses = []
                clip_rep_losses = []
                for k in self.keys:
                    a_g = adv_gl[k][0][i:i+1]
                    t_g = tgt_gl[k][0][i:i+1]
                    s_g = src_gl[k][0]

                    g_loss = _sinkhorn_ot(a_g, t_g.squeeze(0),
                                          eps=C.FOA_SINKHORN_EPS, iters=C.FOA_SINKHORN_ITERS,
                                          tol=C.FOA_SINKHORN_TOL)
                    l_loss = _sinkhorn_ot(kmeans_tokens(adv_gl[k][1][i]), kmeans_tokens(tgt_gl[k][1][i]),
                                          eps=C.FOA_SINKHORN_EPS, iters=C.FOA_SINKHORN_ITERS,
                                          tol=C.FOA_SINKHORN_TOL)
                    rep_loss = torch.sum(a_g * s_g, dim=-1).mean()
                    clip_tgt_losses.append(g_loss + clip_local_weight * l_loss)
                    clip_rep_losses.append(rep_loss)

                loss_clip = sum(clip_tgt_losses[j] - beta_rep * clip_rep_losses[j]
                                for j in range(len(self.keys))) / len(self.keys)

                # Branch 2: DINOv2
                d_a_g = adv_gl["Dino"][0][i:i+1]
                d_t_g = tgt_gl["Dino"][0][i:i+1]
                d_s_g = src_gl["Dino"][0]

                d_g_loss = _sinkhorn_ot(d_a_g, d_t_g.squeeze(0),
                                        eps=C.FOA_SINKHORN_EPS, iters=C.FOA_SINKHORN_ITERS,
                                        tol=C.FOA_SINKHORN_TOL)
                d_l_loss = _sinkhorn_ot(kmeans_tokens(adv_gl["Dino"][1][i]), kmeans_tokens(tgt_gl["Dino"][1][i]),
                                        eps=C.FOA_SINKHORN_EPS, iters=C.FOA_SINKHORN_ITERS,
                                        tol=C.FOA_SINKHORN_TOL)
                d_rep_loss = torch.sum(d_a_g * d_s_g, dim=-1).mean()
                loss_dino = (d_g_loss + dino_local_weight * d_l_loss) - beta_rep * d_rep_loss

                crop_losses.append(0.5 * loss_clip + 0.5 * loss_dino)

            loss_unified = 0.5 * (crop_losses[0] + crop_losses[1])

            # Gradient and momentum update
            grad = torch.autograd.grad(loss_unified, delta, create_graph=False)[0]
            with torch.no_grad():
                grad_norm = grad / (torch.mean(torch.abs(grad)) + 1e-8)
                momentum = momentum_decay * momentum + grad_norm
                delta = torch.clamp(delta + alpha * momentum.sign(), -epsilon, epsilon)
            losses.append(float(loss_unified.detach()))

        adv = torch.clamp((x0 + delta.detach()) / 255.0, 0.0, 1.0)
        return {"adv": adv.detach()[0], "losses": losses,
                "linf": float(delta.detach().abs().max()) / 255.0,
                "final_s_pos": float(losses[-1]),
                "final_s_neg": float(d_rep_loss.detach())}


    # ----- M-Attack-V2 (MCA + Momentum Replay) -----
    def attack_mattack_v2(self, source_pil: Image.Image, target_pil: Image.Image,
                          steps: int = C.OFFICIAL_STEPS, seed: int = 20260827,
                          alpha: float = 1.0, epsilon: float = 16.0,
                          multi_pass_num: int = 4, momentum_decay: float = 0.9) -> dict:
        """M-Attack-V2: Multi-Crop Alignment (MCA) + Patch Momentum (Momentum Replay)."""
        x0 = pil_to255(source_pil).to(self.device)
        xt = pil_to255(target_pil).to(self.device)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        delta = torch.zeros_like(x0)
        momentum = torch.zeros_like(x0)
        losses = []

        for _ in range(steps):
            accum_grad = torch.zeros_like(x0)
            step_loss = 0.0
            for _ in range(multi_pass_num):
                delta.requires_grad_(True)
                adv_crop = _rrc_crop(x0 + delta, gen)
                with torch.no_grad():
                    tgt_crop = _rrc_crop(xt, gen)
                    gt = self._global_projected(tgt_crop)
                feats = self._global_projected(adv_crop)
                pass_loss = sum(torch.sum(feats[k] * gt[k], dim=-1).mean() for k in self.keys) / len(self.keys)
                g = torch.autograd.grad(pass_loss, delta, create_graph=False)[0]
                accum_grad += g / multi_pass_num
                step_loss += float(pass_loss.detach()) / multi_pass_num
                delta = delta.detach()

            with torch.no_grad():
                grad_norm = accum_grad / (torch.mean(torch.abs(accum_grad)) + 1e-8)
                momentum = momentum_decay * momentum + grad_norm
                delta = torch.clamp(delta + alpha * momentum.sign(), -epsilon, epsilon)
            losses.append(step_loss)

        adv = torch.clamp((x0 + delta.detach()) / 255.0, 0.0, 1.0)
        return {"adv": adv.detach()[0], "losses": losses,
                "linf": float(delta.detach().abs().max()) / 255.0,
                "final_s_pos": float(losses[-1]), "final_s_neg": 0.0}

    # ----- Baseline: AttackVLM (standard projected CLS PGD) -----
    def attack_attackvlm(self, source_pil: Image.Image, target_pil: Image.Image,
                         steps: int = C.OFFICIAL_STEPS, seed: int = 20260827,
                         alpha: float = 1.0, epsilon: float = 16.0) -> dict:
        """AttackVLM: PGD cosine ascent in projected vision feature space without crop augmentation."""
        x0 = pil_to255(source_pil).to(self.device)
        xt = pil_to255(target_pil).to(self.device)
        delta = torch.zeros_like(x0)
        losses = []
        with torch.no_grad():
            tgt_px = _norm255(xt, C.ATTACK_RES)
            gt = {k: F.normalize(self.bank.clips[k].get_image_features(pixel_values=tgt_px).float(), dim=-1)
                  for k in self.keys}

        for _ in range(steps):
            delta.requires_grad_(True)
            adv_px = _norm255(x0 + delta, C.ATTACK_RES)
            feats = {k: F.normalize(self.bank.clips[k].get_image_features(pixel_values=adv_px).float(), dim=-1)
                     for k in self.keys}
            loss = sum(torch.sum(feats[k] * gt[k], dim=-1).mean() for k in self.keys) / len(self.keys)
            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
            with torch.no_grad():
                delta = torch.clamp(delta + alpha * grad.sign(), -epsilon, epsilon)
            losses.append(float(loss.detach()))

        adv = torch.clamp((x0 + delta.detach()) / 255.0, 0.0, 1.0)
        return {"adv": adv.detach()[0], "losses": losses,
                "linf": float(delta.detach().abs().max()) / 255.0,
                "final_s_pos": float(losses[-1]), "final_s_neg": 0.0}

    # ----- Baseline: AnyAttack (CVPR 2025 official self-supervised generative decoder) -----
    def attack_anyattack(self, source_pil: Image.Image, target_pil: Image.Image,
                         steps: int = C.OFFICIAL_STEPS, seed: int = 20260827,
                         alpha: float = 1.0, epsilon: float = 16.0) -> dict:
        """AnyAttack Official (CVPR 2025): Self-supervised Generative Decoder.
        Generates targeted adversarial noise via single-pass forward propagation through
        the official pre-trained Decoder fine-tuned on Flickr30k.
        """
        if not hasattr(self, "_anyattack_dec"):
            from .models import load_anyattack_decoder
            self._anyattack_dec = load_anyattack_decoder(device=self.device, variant="cos")

        x0 = pil_to255(source_pil).to(self.device)
        xt = pil_to255(target_pil).to(self.device)

        clip_b32 = self.bank.clips["B32"]
        with torch.no_grad():
            px_t = _norm255(xt, C.ATTACK_RES)
            t_emb = clip_b32.get_image_features(pixel_values=px_t)
            t_emb = F.normalize(t_emb.float(), dim=-1)

            origin_noise = self._anyattack_dec(t_emb)  # [1, 3, 224, 224]
            noise_01 = torch.clamp(origin_noise, -epsilon / 255.0, epsilon / 255.0)
            clean_01 = x0 / 255.0
            adv = torch.clamp(clean_01 + noise_01, 0.0, 1.0)
            linf = float(noise_01.abs().max())

        return {"adv": adv[0], "losses": [0.0],
                "linf": linf,
                "final_s_pos": 0.0, "final_s_neg": 0.0}

    # ----- Ablation 1: DINOv2 only -----
    def attack_foa_dino_only(self, source_pil: Image.Image, target_pil: Image.Image,
                             steps: int = C.OFFICIAL_STEPS, seed: int = 20260827) -> dict:
        """Ablation: FOA global-local OT using DINOv2 branch only (K=1)."""
        return self._attack_foa(source_pil, target_pil, ["Dino"], steps, seed)

    # ----- Ablation 2: w/o Local OT (Global OT only) -----
    def attack_foa_no_local(self, source_pil: Image.Image, target_pil: Image.Image,
                            steps: int = C.OFFICIAL_STEPS, seed: int = 20260827) -> dict:
        """Ablation: Global CLS OT only (local weight = 0)."""
        old_w = C.FOA_LOCAL_WEIGHT
        try:
            C.FOA_LOCAL_WEIGHT = 0.0
            return self._attack_foa(source_pil, target_pil, self.keys_v2, steps, seed)
        finally:
            C.FOA_LOCAL_WEIGHT = old_w


    # ----- Ablation 3: step-budget efficiency (100 steps) -----
    def attack_foa_v2_s100(self, source_pil: Image.Image, target_pil: Image.Image,
                           steps: int = C.OFFICIAL_STEPS, seed: int = 20260827) -> dict:
        """Ablation: FOA-Attack v2 at a 100-step budget (efficiency row).
        The CLI steps value is ignored; always runs exactly 100 steps."""
        return self.attack_foa_v2(source_pil, target_pil, steps=100, seed=seed)


ARMS = [
    "foa_v2",
    "foa_official",
    "mattack_v2",
    "mattack_official",
    "anyattack",
    "attackvlm",
    "foa_dino_only",
    "foa_no_local",
    "foa_v2_s100",
    "croa_tuned",
]

ARM_LABELS = {
    "foa_v2": "CROA / FOA-v2 (Ours: official FOA + DINOv2 branch)",
    "foa_official": "FOA-Attack (NeurIPS 2025, official)",
    "mattack_v2": "M-Attack-V2 (2026, official MCA+Momentum)",
    "mattack_official": "M-Attack (NeurIPS 2025, official)",
    "anyattack": "AnyAttack (CVPR 2025, official generative)",
    "attackvlm": "AttackVLM (CVPR 2024)",
    "foa_dino_only": "Ablation: DINOv2-only",
    "foa_no_local": "Ablation: Global OT only (w/o Local OT)",
    "foa_v2_s100": "Ablation: FOA-Attack v2 @ 100 steps",
    "croa_tuned": "CROA-Tuned (4-pass MCA, scale 0.35-1.0, beta=0.5)",
}

