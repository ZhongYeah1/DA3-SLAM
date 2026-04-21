"""
DA3Wrapper: Adapter that wraps Depth-Anything-3 to provide a VGGT-compatible
interface for the SLAM solver (solver.py).

The solver calls:
    predictions = model(images)                                # N-frame inference
    predictions_lc = model(lc_frames, compute_similarity=True) # 2-frame loop closure

and expects a dict with keys:
    depth (N,H,W,1), depth_conf (N,H,W), extrinsic (N,3,4),
    intrinsic (N,3,3), images (N,3,H,W), and optionally image_match_ratio.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


from depth_anything_3.api import DepthAnything3


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def unproject_depth_to_points(
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Unproject a depth map to camera-space 3-D points.

    Returns points in each frame's **camera coordinate system** (not world).
    This matches VGGT-SLAM's convention where the submap stores camera-space
    points and the graph homography later transforms them to world space.

    Args:
        depth: (N, H, W) metric depth.
        extrinsics: (N, 3, 4) world-to-camera -- unused, kept for API compat.
        intrinsics: (N, 3, 3) camera intrinsic matrices.

    Returns:
        (N, H, W, 3) array of camera-space 3-D points.
    """
    N, H, W = depth.shape

    # Build pixel grid (H, W)
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # both (H, W)

    points_cam = np.empty((N, H, W, 3), dtype=np.float32)
    for i in range(N):
        fx = intrinsics[i, 0, 0]
        fy = intrinsics[i, 1, 1]
        cx = intrinsics[i, 0, 2]
        cy = intrinsics[i, 1, 2]

        # Camera-space coordinates (pinhole model, no extrinsic transform)
        z = depth[i]  # (H, W)
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        points_cam[i] = np.stack([x, y, z], axis=-1)  # (H, W, 3)

    return points_cam


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class DA3Wrapper(nn.Module):
    """Wraps Depth-Anything-3 with the dict-based interface expected by
    VGGT-SLAM's solver.py.
    """

    def __init__(
        self,
        model_name: str = "da3-small",
        device: str = "cuda",
        process_res: int = 504,
        target_layer: int = -3,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.process_res = process_res
        self.target_layer = target_layer

        # Map short model names to HuggingFace repo IDs
        HF_MODEL_MAP = {
            "da3-small": "depth-anything/DA3-SMALL",
            "da3-base": "depth-anything/DA3-BASE",
            "da3-large": "depth-anything/DA3-LARGE-1.1",
            "da3-giant": "depth-anything/DA3-GIANT-1.1",
            "da3metric-large": "depth-anything/DA3METRIC-LARGE",
            "da3mono-large": "depth-anything/DA3MONO-LARGE",
            "da3nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        }
        hf_repo = HF_MODEL_MAP.get(model_name, model_name)

        # Load DA3 model WITH pretrained weights
        print(f"Loading DA3 pretrained weights from {hf_repo}...")
        self.da3 = DepthAnything3.from_pretrained(hf_repo)
        self.da3.to(self.device)
        self.da3.eval()

        # Resolve the absolute block index for the target layer
        blocks = self._get_backbone_blocks()
        self._num_blocks = len(blocks)
        self._abs_target_layer = (
            self.target_layer if self.target_layer >= 0
            else self._num_blocks + self.target_layer
        )

        # Hook bookkeeping for attention-based similarity
        self._hook_handle = None
        self._captured_k: Optional[torch.Tensor] = None
        self._captured_q: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Backbone access
    # ------------------------------------------------------------------

    def _get_backbone_blocks(self) -> nn.ModuleList:
        """Return the list of transformer blocks from the DinoV2 backbone.

        Plain DA3 nets expose ``model.backbone``; NESTED nets (e.g.
        ``da3nested-giant-large``) wrap an any-view submodel at ``model.da3``,
        so the backbone lives one level deeper.
        """
        m = self.da3.model
        if hasattr(m, "backbone"):
            return m.backbone.pretrained.blocks
        return m.da3.backbone.pretrained.blocks

    # ------------------------------------------------------------------
    # Similarity hook management
    # ------------------------------------------------------------------

    def _register_similarity_hook(self) -> None:
        """Register a forward hook on the target block's attention module to
        capture K and Q tensors for VGGT-style similarity computation.

        Hooks into ``blocks[target_layer].attn`` (the DinoV2 ``Attention``
        module).  Inside the hook we recompute QKV from the attention input
        and store K and Q with shape ``(B, num_heads, N, head_dim)``.
        """
        blocks = self._get_backbone_blocks()
        target_attn = blocks[self._abs_target_layer].attn

        def hook_fn(module, input, output):
            # ``input`` is a tuple; the first element is ``x`` with shape
            # (B, N, C) -- the norm1-ed token sequence passed to the Attention
            # module.
            x = input[0]  # (B, N, C)
            B, N, C = x.shape
            num_heads = module.num_heads
            head_dim = C // num_heads

            # Compute QKV the same way the Attention module does
            qkv = (
                module.qkv(x)
                .reshape(B, N, 3, num_heads, head_dim)
                .permute(2, 0, 3, 1, 4)
            )  # (3, B, H, N, D)
            q, k = qkv[0], qkv[1]
            q = module.q_norm(q)
            k = module.k_norm(k)

            self._captured_k = k.detach()  # (B, H, N, D)
            self._captured_q = q.detach()  # (B, H, N, D)

        self._hook_handle = target_attn.register_forward_hook(hook_fn)

    def _remove_similarity_hook(self) -> None:
        """Remove a previously registered similarity hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._captured_k = None
        self._captured_q = None

    # ------------------------------------------------------------------
    # Attention-based similarity (VGGT-style)
    # ------------------------------------------------------------------

    def _compute_attention_similarity(self) -> float:
        """Compute VGGT-style attention-based image match ratio from captured
        K and Q tensors.

        Token layout for 2 images in DA3's DinoV2 backbone::

            [CLS_1, patch_1_0 .. patch_1_{T-1}, CLS_2, patch_2_0 .. patch_2_{T-1}]

        ``token_offset = 1`` (skip CLS only -- DA3 has no register tokens).

        Algorithm (adapted from VGGT ``get_similarity``):

        Steps 1-5 follow VGGT exactly:

        1. Select K from image-1 patch tokens only (skip CLS).
        2. Compute attention: all Q tokens attend to image-1 K tokens.
        3. Average across heads.
        4. Split attention into per-frame components.
        5. Normalize frame-2 attention by max self-attention of frame-1.

        Step 6 uses **median** instead of VGGT's mean-top-quarter because
        DA3's DinoV2 backbone (with qk_norm + RoPE) produces heavy-tailed
        attention ratios -- a few tokens can exceed 1000x, making
        mean-top-quarter unstable.  The median is robust to these outliers
        and gives clean separation: overlapping pairs ~0.85+, non-overlapping
        pairs ~0.01-0.05.
        """
        k = self._captured_k  # (B, H, N_total, D)
        q = self._captured_q  # (B, H, N_total, D)

        if k is None or q is None:
            print("WARNING: Attention hook did not fire -- cannot compute similarity")
            return -1.0

        token_offset = 1  # skip CLS token per image
        B, H, N_total, D = q.shape
        if N_total < 4:  # need at least 2 tokens per image
            return -1.0
        tokens_per_img = N_total // 2

        # 1. K from image-1 patch tokens only (skip CLS)
        k_img1 = k[:, :, token_offset:tokens_per_img, :]  # (B, H, T, D)

        # 2. Attention: all Q tokens attend to image-1 K patch tokens
        attn = q @ k_img1.transpose(-2, -1)  # (B, H, N_total, T)
        attn = attn.transpose(-2, -1)  # (B, H, T, N_total)
        attn = attn.softmax(dim=-1)  # normalize across N_total (the sequence Q came from)

        # 3. Average across heads
        attn = attn.mean(dim=1)  # (B, T, N_total)

        # 4. Split into per-frame attention
        attn_to_frame1 = attn[..., :tokens_per_img]  # (B, T, tokens_per_img)
        attn_to_frame2 = attn[..., tokens_per_img:]  # (B, T, tokens_per_img)

        # 5. Normalize: divide frame-2 attention by max frame-1 self-attention
        max_per_token = attn_to_frame1.max(dim=-1)[0]  # (B, T)
        attn_second_normalized = attn_to_frame2 / (
            max_per_token.unsqueeze(-1) + 1e-8
        )  # (B, T, tokens_per_img)

        # 6. Aggregate: max across image-1 patch tokens (dim=1), then median
        #    (robust to the heavy-tailed ratio distribution in DA3's backbone)
        ratio = attn_second_normalized.max(dim=1)[0]  # (B, tokens_per_img)
        ratio_np = ratio.float().detach().cpu().numpy()

        return float(np.median(ratio_np))

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def load_and_preprocess(self, image_paths: List[str]) -> torch.Tensor:
        """Use DA3's input_processor to load and preprocess images.

        Args:
            image_paths: List of file paths to images.

        Returns:
            Tensor of shape (1, N, 3, H, W) on the model device.
        """
        imgs_cpu, _, _ = self.da3.input_processor(
            image_paths,
            process_res=self.process_res,
        )
        # imgs_cpu is (N, 3, H, W); add batch dim and move to device
        return imgs_cpu.unsqueeze(0).float().to(self.device)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    # ImageNet normalization constants (must match DA3's InputProcessor)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _denormalize_images(self, imgs_normalized: np.ndarray) -> np.ndarray:
        """Convert ImageNet-normalized (N,3,H,W) images back to [0,1] range.

        Args:
            imgs_normalized: (N, 3, H, W) ImageNet-normalized numpy array.

        Returns:
            (N, 3, H, W) numpy array in [0, 1] range.
        """
        # mean/std are (3,) → broadcast as (1, 3, 1, 1)
        mean = self.IMAGENET_MEAN[None, :, None, None]
        std = self.IMAGENET_STD[None, :, None, None]
        return np.clip(imgs_normalized * std + mean, 0.0, 1.0)

    def __call__(
        self,
        images: Union[List[str], torch.Tensor],
        compute_similarity: bool = False,
    ) -> dict:
        """Run DA3 inference and return a VGGT-SLAM compatible dict.

        Args:
            images: Either a list of file paths or a tensor of shape (N, 3, H, W).
                When file paths, images are preprocessed internally (ImageNet
                normalization + patch-aligned resize).  When a tensor, it must
                already be preprocessed (ImageNet-normalized).
            compute_similarity: If True, also compute image_match_ratio via
                VGGT-style attention-based similarity (used for loop closure).

        Returns:
            Dict with keys:
                depth         – (N, H, W, 1) numpy
                depth_conf    – (N, H, W) numpy
                extrinsic     – (N, 3, 4) numpy  (w2c convention)
                intrinsic     – (N, 3, 3) numpy
                images        – (N, 3, H, W) numpy  **[0,1] range** (denormalized)
                preprocessed  – (N, 3, H, W) torch tensor on device (ImageNet-normalized)
            and optionally:
                image_match_ratio – float
        """
        # ----- prepare input tensor (1, N, 3, H, W) -----------------------
        if isinstance(images, list):
            imgs_tensor = self.load_and_preprocess(images)
        elif isinstance(images, torch.Tensor):
            if images.dim() == 4:
                # (N, 3, H, W) -> (1, N, 3, H, W)
                imgs_tensor = images.unsqueeze(0).to(self.device)
            else:
                imgs_tensor = images.to(self.device)
        else:
            raise TypeError(f"Unsupported images type: {type(images)}")

        # ----- optionally register similarity hook -------------------------
        if compute_similarity:
            self._register_similarity_hook()

        try:
            # ----- forward pass --------------------------------------------
            # Use ref_view_strategy="first" so the first input frame is the
            # reference and its predicted extrinsic is identity.
            raw_output = self.da3.forward(
                imgs_tensor, export_feat_layers=[], ref_view_strategy="first",
            )

            # ----- convert to Prediction dataclass -------------------------
            prediction = self.da3.output_processor(raw_output)

            # ----- build result dict ---------------------------------------
            # depth: Prediction gives (N, H, W), solver expects (N, H, W, 1)
            depth = prediction.depth  # (N, H, W) numpy
            depth = depth[..., np.newaxis]  # (N, H, W, 1)

            # depth_conf: (N, H, W)
            depth_conf = prediction.conf
            if depth_conf is None:
                depth_conf = np.ones(prediction.depth.shape, dtype=np.float32)

            # extrinsic: Prediction gives (N, 3, 4) w2c, solver expects (N, 3, 4).
            # With ref_view_strategy="first", frame 0 is already identity.
            extrinsics = prediction.extrinsics  # (N, 3, 4) or None
            if extrinsics is not None:
                extrinsic = extrinsics  # (N, 3, 4)
            else:
                N = depth.shape[0]
                extrinsic = np.tile(np.eye(4)[:3, :], (N, 1, 1)).astype(np.float32)

            # intrinsic: (N, 3, 3)
            intrinsic = prediction.intrinsics
            if intrinsic is None:
                N, H, W = prediction.depth.shape
                intrinsic = np.zeros((N, 3, 3), dtype=np.float32)
                intrinsic[:, 0, 0] = W  # fx ~ W as a rough default
                intrinsic[:, 1, 1] = H  # fy ~ H
                intrinsic[:, 0, 2] = W / 2.0
                intrinsic[:, 1, 2] = H / 2.0
                intrinsic[:, 2, 2] = 1.0

            # images: denormalize from ImageNet-normalized to [0,1] for color extraction
            imgs_normalized_np = imgs_tensor.squeeze(0).cpu().float().numpy()  # (N, 3, H, W)
            images_01 = self._denormalize_images(imgs_normalized_np)  # (N, 3, H, W) in [0,1]

            result = {
                "depth": depth,
                "depth_conf": depth_conf,
                "extrinsic": extrinsic,
                "intrinsic": intrinsic,
                "images": images_01,                       # [0,1] range for color extraction
                "preprocessed": imgs_tensor.squeeze(0),    # ImageNet-normalized tensor on device
            }

            # ----- similarity (loop closure) -------------------------------
            if compute_similarity:
                ratio = self._compute_attention_similarity()
                result["image_match_ratio"] = ratio

        finally:
            # Always clean up the hook
            if compute_similarity:
                self._remove_similarity_hook()

        return result

    def forward_with_poses(
        self,
        images: Union[List[str], torch.Tensor],
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
    ) -> dict:
        """Run DA3 inference conditioned on known camera poses.

        Used by the refinement pass: GTSAM-optimized poses are provided
        as conditioning so DA3 produces depth consistent with those poses.

        Args:
            images: List of file paths or (N, 3, H, W) tensor.
            extrinsics: (N, 4, 4) w2c numpy -- e.g. from GTSAM decomposition.
            intrinsics: (N, 3, 3) numpy -- from pass 1.

        Returns:
            Same dict format as __call__.
        """
        # ----- prepare image tensor (1, N, 3, H, W) -----------------------
        if isinstance(images, list):
            imgs_tensor = self.load_and_preprocess(images)
        elif isinstance(images, torch.Tensor):
            if images.dim() == 4:
                imgs_tensor = images.unsqueeze(0).to(self.device)
            else:
                imgs_tensor = images.to(self.device)
        else:
            raise TypeError(f"Unsupported images type: {type(images)}")

        # ----- normalize extrinsics (mirror DA3's _normalize_extrinsics) ---
        ext_t = torch.from_numpy(extrinsics).float().unsqueeze(0).to(self.device)  # (1, N, 4, 4)
        int_t = torch.from_numpy(intrinsics).float().unsqueeze(0).to(self.device)  # (1, N, 3, 3)

        # Normalize: first frame -> identity, scale by median camera distance
        from depth_anything_3.utils.geometry import affine_inverse
        # transform = affine_inverse(ext_t[:, :1])       # inv of first frame
        transform = torch.inverse(ext_t[:, :1])
        ext_norm = ext_t @ transform                     # first frame -> I
        # c2ws = affine_inverse(ext_norm)
        c2ws = torch.inverse(ext_norm)
        translations = c2ws[..., :3, 3]
        dists = translations.norm(dim=-1)
        median_dist = torch.clamp(torch.median(dists), min=1e-1)
        ext_norm[..., :3, 3] = ext_norm[..., :3, 3] / median_dist

        # ----- forward pass with pose conditioning -------------------------
        with torch.no_grad():
            raw_output = self.da3.forward(
                imgs_tensor,
                extrinsics=ext_norm,
                intrinsics=int_t,
                export_feat_layers=[],
                ref_view_strategy="first",
            )

        prediction = self.da3.output_processor(raw_output)

        # ----- build result dict (identical to __call__) -------------------
        # Do NOT call _align_to_input_extrinsics_intrinsics -- that replaces
        # DA3's predicted extrinsics/intrinsics with the INPUT values and
        # rescales depth by the Umeyama scale factor, creating a scale
        # mismatch between pass 1 and pass 2. Instead, return DA3's raw
        # predictions processed exactly the same way as __call__.

        # depth: Prediction gives (N, H, W), solver expects (N, H, W, 1)
        depth = prediction.depth  # (N, H, W) numpy
        depth = depth[..., np.newaxis]  # (N, H, W, 1)

        # depth_conf: (N, H, W)
        depth_conf = prediction.conf
        if depth_conf is None:
            depth_conf = np.ones(prediction.depth.shape, dtype=np.float32)

        # extrinsic: Prediction gives (N, 3, 4) w2c, solver expects (N, 3, 4).
        # With ref_view_strategy="first", frame 0 is already identity.
        extrinsics_pred = prediction.extrinsics  # (N, 3, 4) or None
        if extrinsics_pred is not None:
            extrinsic = extrinsics_pred  # (N, 3, 4)
        else:
            N = depth.shape[0]
            extrinsic = np.tile(np.eye(4)[:3, :], (N, 1, 1)).astype(np.float32)

        # intrinsic: (N, 3, 3)
        intrinsic = prediction.intrinsics
        if intrinsic is None:
            N, H, W = prediction.depth.shape
            intrinsic = np.zeros((N, 3, 3), dtype=np.float32)
            intrinsic[:, 0, 0] = W  # fx ~ W as a rough default
            intrinsic[:, 1, 1] = H  # fy ~ H
            intrinsic[:, 0, 2] = W / 2.0
            intrinsic[:, 1, 2] = H / 2.0
            intrinsic[:, 2, 2] = 1.0

        # images: denormalize from ImageNet-normalized to [0,1] for color extraction
        imgs_normalized_np = imgs_tensor.squeeze(0).cpu().float().numpy()  # (N, 3, H, W)
        images_01 = self._denormalize_images(imgs_normalized_np)  # (N, 3, H, W) in [0,1]

        return {
            "depth": depth,
            "depth_conf": depth_conf,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "images": images_01,                       # [0,1] range for color extraction
            "preprocessed": imgs_tensor.squeeze(0),    # ImageNet-normalized tensor on device
        }
