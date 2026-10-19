"""Frozen DINOv3 features with camera-aligned projection for Pixal3D inference."""
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms


def project_points_to_image_batch(
    points_3d: torch.Tensor,
    transform_matrix: torch.Tensor,
    camera_angle_x: torch.Tensor,
    resolution: int = 518
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D image coordinates (batch processing).

    Args:
        points_3d: torch.Tensor, shape [N, 3] or [B, N, 3], 3D point coordinates (in [-1, 1] range)
        transform_matrix: torch.Tensor, shape [B, 4, 4], camera transformation matrix
        camera_angle_x: torch.Tensor, shape [B], horizontal field of view angle (radians)
        resolution: int, image resolution, default 518

    Returns:
        points_2d: torch.Tensor, shape [B, N, 2], image coordinates [x, y]
        depth: torch.Tensor, shape [B, N], depth values
        valid_mask: torch.Tensor, shape [B, N], mask for points within view
    """
    device = points_3d.device
    B = transform_matrix.shape[0]

    # Ensure inputs are torch.Tensor on correct device
    if not isinstance(transform_matrix, torch.Tensor):
        transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32, device=device)
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.tensor(points_3d, dtype=torch.float32, device=device)
    if not isinstance(camera_angle_x, torch.Tensor):
        camera_angle_x = torch.tensor(camera_angle_x, dtype=torch.float32, device=device)

    # Expand points_3d to batch dimension: [N, 3] -> [B, N, 3]
    if points_3d.dim() == 2:
        points_3d_batch = points_3d.unsqueeze(0).expand(B, -1, -1)
    else:
        points_3d_batch = points_3d
    N = points_3d_batch.shape[1]

    # Add homogeneous coordinates: [B, N, 3] -> [B, N, 4]
    ones = torch.ones(B, N, 1, device=device, dtype=points_3d_batch.dtype)
    points_homogeneous = torch.cat([points_3d_batch, ones], dim=-1)  # [B, N, 4]

    # Compute world to camera transformation matrix
    world_to_camera = torch.linalg.inv(transform_matrix.float()).to(transform_matrix.dtype)  # linalg.inv requires fp32+

    # Batch transform to camera coordinate system: [B, N, 4] @ [B, 4, 4]^T -> [B, N, 3]
    points_camera = torch.bmm(points_homogeneous, world_to_camera.transpose(-2, -1))[..., :3]  # [B, N, 3]

    # Extract camera coordinates
    x_cam = points_camera[..., 0]  # [B, N]
    y_cam = points_camera[..., 1]  # [B, N]
    z_cam = points_camera[..., 2]  # [B, N]

    # Depth value (Z value in camera coordinate system, note Blender camera faces -Z direction)
    depth = -z_cam  # [B, N]

    # Compute camera intrinsics (batch processing)
    sensor_width = 32.0  # mm
    focal_length = 16.0 / torch.tan(camera_angle_x / 2.0)  # [B]
    focal_length_pixels = focal_length * resolution / sensor_width  # [B]

    # Expand focal_length_pixels dimension for broadcasting: [B] -> [B, 1]
    focal_length_pixels = focal_length_pixels.unsqueeze(1)  # [B, 1]

    # Perspective projection to NDC coordinates
    x_ndc = focal_length_pixels * x_cam / (-z_cam + 1e-8)  # [B, N]
    y_ndc = focal_length_pixels * y_cam / (-z_cam + 1e-8)  # [B, N]

    # Convert to image coordinates (pixel coordinates)
    x_pixel = x_ndc + resolution / 2.0  # [B, N]
    y_pixel = -y_ndc + resolution / 2.0  # [B, N], flip Y axis

    # Create validity mask (points within image range and in front of camera)
    valid_mask = (
        (x_pixel >= 0) & (x_pixel < resolution) &
        (y_pixel >= 0) & (y_pixel < resolution) &
        (depth > 0)  # In front of camera
    )  # [B, N]

    points_2d = torch.stack([x_pixel, y_pixel], dim=-1)  # [B, N, 2]

    return points_2d, depth, valid_mask

def sample_features(fmap: torch.Tensor, queries_ndc: torch.Tensor) -> torch.Tensor:
    """
    Sample features from feature map at specified NDC coordinates.

    Args:
        fmap: torch.Tensor, shape [B, C, H, W], feature map
        queries_ndc: torch.Tensor, shape [B, K, 2], normalized device coordinates

    Returns:
        torch.Tensor, shape [B, C, K], sampled features
    """
    B, C, H, W = fmap.shape
    Bq, K, _ = queries_ndc.shape
    assert Bq == B, "Batch size mismatch"

    # grid_sample requires (B, out_h, out_w, 2), here we want K points -> out_h=K, out_w=1
    grid = queries_ndc.view(B, K, 1, 2)  # (B, K, 1, 2)

    # Bilinear interpolation, align_corners=False (consistent with [-1,1] pixel center convention)
    feat = F.grid_sample(
        fmap, grid, mode='bilinear',
        align_corners=False, padding_mode='border'  # border avoids out-of-bound becoming 0
    )  # (B, C, K, 1)

    return feat.squeeze(-1)  # (B, C, K)

class ProjGrid(nn.Module):
    """
    3D Grid Projection Module.

    Projects a 3D grid of points to 2D image coordinates and samples features
    from the image feature map at those locations.

    This is the core module for view-aligned feature extraction.
    """
    def __init__(self, grid_resolution: int = 16, image_resolution: int = 518):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.image_resolution = image_resolution

        # Create 3D grid points
        one_dim = torch.linspace(-1, 1, grid_resolution)
        x, y, z = torch.meshgrid(one_dim, one_dim, one_dim, indexing='ij')
        grid_points = torch.stack((x, y, z), dim=-1)

        # Rotation matrix to align with Blender coordinate system
        rotation_matrix = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ])
        grid_points = torch.matmul(grid_points, rotation_matrix.T)
        grid_points = grid_points.reshape(-1, 3)
        self.register_buffer('grid_points', grid_points)  # [R³, 3]

        # Default front view transformation matrix
        front_view_transform_matrix = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.register_buffer("front_view_transform_matrix", front_view_transform_matrix)

    def forward(
        self,
        features_map: torch.Tensor,
        camera_angle_x: torch.Tensor,
        distance: torch.Tensor,
        mesh_scale: torch.Tensor,
        transform_matrix: Optional[torch.Tensor] = None,
        BHWC: bool = True
    ) -> torch.Tensor:
        """
        Project 3D grid points to image and sample features.

        Args:
            features_map: Feature map, shape [B, H, W, C] if BHWC else [B, C, H, W]
            camera_angle_x: Camera FOV angle, shape [B]
            distance: Camera distance, shape [B]
            mesh_scale: Mesh scale factor, shape [B]
            transform_matrix: Optional camera transform matrix, shape [B, 4, 4]
            BHWC: Whether features_map is in BHWC format

        Returns:
            Projected features, shape [B, grid_resolution³, C]
        """
        if BHWC:
            B, H, W, C = features_map.shape
        else:
            B, C, H, W = features_map.shape

        grid_points = self.grid_points
        grid_points = grid_points.expand(B, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2  # Scale alignment
        assert transform_matrix is None, "transform_matrix is not None"
        if transform_matrix is None:
            transform_matrix = self.front_view_transform_matrix
            transform_matrix = transform_matrix.expand(B, -1, -1).clone()
            transform_matrix[:, 1, 3] = -distance  # Set camera distance

        # Project to image coordinates (simulate Blender projection)
        image_points, depth, valid_mask = project_points_to_image_batch(
            grid_points, transform_matrix, camera_angle_x, self.image_resolution
        )

        # Normalize to [-1, 1] for grid_sample
        image_points_norm = (image_points + 0.5) / self.image_resolution * 2 - 1

        if BHWC:
            features_map = features_map.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Sample features from DINOv3 patch feature map
        x = sample_features(features_map, image_points_norm)  # [B, C, K]
        x = x.permute(0, 2, 1)  # [B, K, C]

        return x


class DinoV3ProjFeatureExtractor(nn.Module):
    """
    DINOv3 Feature Extractor with View-Aligned Projection.

    This extractor produces both:
    1. Global features (CLS token + register tokens) in embed_dim
    2. View-aligned projected features (3D grid projected to 2D and sampled)
       - Without NAF: [B, R³, embed_dim]
       - With NAF:    [B, R³, embed_dim * 2]  (concat of lr and hr features)

    NOTE: proj_linear has been moved to per-block ProjectAttention / SparseProjectAttention.
    This module now outputs raw DINOv3 features for proj (optionally concatenated with NAF-upsampled features).

    Args:
        model_name: Name of the pretrained DINOv3 model
        image_size: Input image size (default: 512)
        grid_resolution: Resolution of the 3D projection grid (default: 16)
        use_naf_upsample: Whether to use NAF to upsample features (default: False)
        naf_target_size: Target spatial size for NAF upsampling (default: [128, 128])
    """
    def __init__(
        self,
        model_name: str,
        image_size: int = 512,
        grid_resolution: int = 16,
        use_naf_upsample: bool = False,
        naf_target_size: Optional[List[int]] = None,
        cls_only: bool = False,
        cls_n_tokens: int = 1,
        full_tokens: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self.grid_resolution = grid_resolution
        self.use_naf_upsample = use_naf_upsample
        self.cls_only = cls_only
        self.cls_n_tokens = cls_n_tokens
        self.full_tokens = full_tokens
        if naf_target_size is None:
            self.naf_target_size = (128, 128)
        elif isinstance(naf_target_size, int):
            self.naf_target_size = (naf_target_size, naf_target_size)
        else:
            self.naf_target_size = tuple(naf_target_size)

        # Load DINOv3 model (frozen, no trainable params in this module)
        from transformers import DINOv3ViTModel

        self.model = DINOv3ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.model.requires_grad_(False)

        # Image transform (only normalize, no resize - assume already resized)
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Get patch info
        self.patch_size = self.model.config.patch_size
        self.patch_number = image_size // self.patch_size
        self.embed_dim = self.model.config.hidden_size

        # Projection grid for view-aligned features
        self.proj_grid = ProjGrid(
            grid_resolution=grid_resolution,
            image_resolution=image_size
        )

        # NAF upsampler (frozen, no trainable params)
        self.naf_model = None  # Lazy-loaded on first use to avoid import if not needed

        # proj_channels: the output dimension of proj features
        # Without NAF: embed_dim (e.g. 1024)
        # With NAF: embed_dim * 2 (e.g. 2048, concat of lr and hr)
        self.proj_channels = self.embed_dim * 2 if use_naf_upsample else self.embed_dim

        # NOTE: proj_linear removed — now lives in each denoiser block's ProjectAttention

    def _load_naf(self):
        """Lazy-load pretrained NAF model."""
        if self.naf_model is None:
            import torch.hub
            device = next(self.model.parameters()).device
            self.naf_model = torch.hub.load(
                "valeoai/NAF", "naf", pretrained=True, device=device, trust_repo=True
            )
            self.naf_model.eval()
            self.naf_model.requires_grad_(False)

    def to(self, device):
        super().to(device)
        self.model.to(device)
        self.proj_grid.to(device)
        if self.naf_model is not None:
            self.naf_model.to(device)
        return self

    def cuda(self):
        super().cuda()
        self.model.cuda()
        self.proj_grid.cuda()
        if self.naf_model is not None:
            self.naf_model.cuda()
        return self

    def cpu(self):
        super().cpu()
        self.model.cpu()
        self.proj_grid.cpu()
        if self.naf_model is not None:
            self.naf_model.cpu()
        return self

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features using DINOv3."""
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for layer_module in self.model.layer:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    def forward(
        self,
        image: Union[torch.Tensor, List[Image.Image]],
        camera_angle_x: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
        mesh_scale: Optional[torch.Tensor] = None,
        transform_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract view-aligned features from the image.

        Args:
            image: Input image tensor [B, C, H, W] or list of PIL images
            camera_angle_x: Camera FOV angle in radians [B]
            distance: Camera distance [B]
            mesh_scale: Mesh scale factor [B]
            transform_matrix: Optional camera transform matrix [B, 4, 4]

        Returns:
            Tuple of (global_features, proj_features):
            - global_features: [B, num_global_tokens, embed_dim]
            - proj_features: [B, grid_resolution³, proj_channels]
              where proj_channels = embed_dim (no NAF) or embed_dim*2 (with NAF)
        """
        # Handle input types
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        B = image.shape[0]

        # Trained-bridge CLS cross-attn mode: skip ProjGrid + NAF (no camera); return global token(s).
        # cls_n_tokens=1 -> CLS only (v4 default); =5 (1 CLS + 4 register) -> faithful warm start of the
        # shipped cross-attn (trained on [CLS + register] global tokens; registers are global, NOT patch).
        # full_tokens=True (v6 patchcross) -> the UNSLICED [CLS|4 reg|patch...] sequence
        # (1029 tokens @ image 512), i.e. the TRELLIS.2 cross-attn context. Lives in this branch
        # because the non-cls path below requires camera params for ProjGrid.
        if getattr(self, 'cls_only', False):
            img = self.transform(image)
            with torch.no_grad():
                z = self.extract_features(img)
            if getattr(self, 'full_tokens', False):
                return z                       # [B, 1+4+patch_number**2, D=1024]
            return z[:, 0:self.cls_n_tokens]   # [B, cls_n_tokens, D=1024]

        # Keep a copy of the unnormalized image for NAF guide
        if self.use_naf_upsample:
            image_for_naf = image.clone()  # [B, 3, H, W], in [0, 1] range

        # Apply transform (ImageNet normalization)
        image = self.transform(image)

        # Extract DINOv3 features (frozen, no gradients)
        with torch.no_grad():
            z = self.extract_features(image)

            # Split into CLS token, register tokens, and patch tokens
            z_clstoken = z[:, 0:1]  # [B, 1, D]
            num_reg = getattr(self.model.config, 'num_register_tokens', 4)
            z_regtokens = z[:, 1:1+num_reg]  # [B, num_reg, D]
            z_patchtokens = z[:, 1+num_reg:]  # [B, num_patches, D]

            # Reshape patch tokens to spatial grid: [B, h, w, D]
            z_patchtokens_spatial = z_patchtokens.reshape(
                B, self.patch_number, self.patch_number, -1
            )  # [B, h, w, D]

            if camera_angle_x is None or distance is None or mesh_scale is None:
                raise ValueError("camera_angle_x, distance, and mesh_scale must be provided")

            # --- Low-resolution branch: sample from DINOv3 patch feature map ---
            z_proj_lr = self.proj_grid(
                z_patchtokens_spatial,
                camera_angle_x,
                distance,
                mesh_scale,
                transform_matrix
            )  # [B, grid_res³, D]

            # --- High-resolution branch (NAF): upsample then sample ---
            if self.use_naf_upsample:
                self._load_naf()
                # NAF expects: guide [B, 3, H, W], lr_features [B, C, h, w], target_size (H', W')
                lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2)  # [B, D, h, w]
                hr_features = self.naf_model(
                    image_for_naf, lr_features_bchw, self.naf_target_size
                )  # [B, D, H', W']

                # Sample from high-res feature map using same projection coordinates
                z_proj_hr = self.proj_grid(
                    hr_features,
                    camera_angle_x,
                    distance,
                    mesh_scale,
                    transform_matrix,
                    BHWC=False  # hr_features is [B, C, H', W']
                )  # [B, grid_res³, D]

                # Concatenate lr and hr: [B, grid_res³, D*2]
                z_proj = torch.cat([z_proj_lr, z_proj_hr], dim=-1)
            else:
                z_proj = z_proj_lr  # [B, grid_res³, D]

            # Combine global tokens
            z_global = torch.cat([z_clstoken, z_regtokens], dim=1)  # [B, 1+num_reg, D]

        # proj_linear has been moved to per-block ProjectAttention
        # z_proj stays in proj_channels, each block will project independently

        return z_global, z_proj
