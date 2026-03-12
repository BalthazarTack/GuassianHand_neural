from dataclasses import dataclass, field
from collections import defaultdict
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from plyfile import PlyData, PlyElement
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


from diff_gaussian_mlp_max_rasterization import (
    Gaussian_MLP_RasterizationSettings,
    Gaussian_MLP_Rasterizer,
)
HAS_MLP_MAX_RASTERIZER = True

# utilities for converting between meshes/gaussians and point clouds
from tgs.utils.graphics_utils import BasicPointCloud


def pcd_from_gaussians(gs: "GaussianModel") -> BasicPointCloud:
    """Produce a BasicPointCloud from a GaussianModel.

    The cloud contains the Gaussian centres with RGB colours extracted from
    the spherical-harmonic coefficients.
    """
    xyz = gs.xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    sh_flat = gs.shs.detach().cpu().numpy().reshape(gs.shs.shape[0], -1)
    if sh_flat.shape[1] >= 3:
        colors = sh_flat[:, :3]
    else:
        colors = np.ones((xyz.shape[0], 3)) * 0.5
    return BasicPointCloud(points=xyz, colors=colors, normals=normals)

def make_default_training_args():
    """Return a namespace filled with the same optimization defaults used by
    splat-the-net's `OptimizationParams` class.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        iterations=10000,
        position_lr_init=0.00016,
        position_lr_final=0.00016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        opacity_volr_lr_init=0.05,
        opacity_volr_lr_final=0.05,
        opacity_volr_lr_delay_mult=0.01,
        opacity_volr_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        opacity_volr_lr=0.03,
        frequencies_mlp_lr=0.001,
        phases_mlp_lr=0.001,
        amplitudes_mlp_lr=0.001,
        scaling_lr=0.005,
        rotation_lr=0.001,
        percent_dense=0.01,
        lambda_dssim=0.2,
        densification_interval=500,
        opacity_reset_interval=3000,
        disable_opacity_reset=True,
        densify_from_iter=1000,
        densify_until_iter=15000,
        densify_grad_threshold=0.0002,
        densify_clone_grad_threshold=1e-4,
        densify_split_grad_threshold=1e-4,
        min_opacity_threshold=0.005,
        init_opacity_volr=50.0,
        random_background=False,
        use_softhresh_density=False,
        lambda_density=0.0,
        lambda_anisotropy=0.001,
        use_regularizer=True,
        regularizer_iter_start=100,
        min_grad_prune=2e-6,
        scale_modifier=1,
        use_positional_grad=False,
        not_use_mlp_grad=False,
        optimize_every=1,
    )
    
from tgs.utils.typing import *
from tgs.utils.base import BaseModule
from tgs.utils.ops import trunc_exp
from tgs.models.networks import MLP
from tgs.utils.ops import scale_tensor
from tgs.models.verts_refinement import vert_valid, vert_pos_refinement
from tgs.models.inter_attn import inter_attn
from tgs.models.self_attn import SelfAttn
from tgs.models.gaussian_model_mlp_HB import GaussianModelMLP_HB
from tgs.utils.general_utils import alpha2density
from livehand.input_encoder import read_mano_uv_obj, save_obj_for_debugging, get_uvd

from einops import rearrange, reduce
import trimesh

inverse_sigmoid = lambda x: np.log(x / (1 - x))

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrix_refine(K: torch.Tensor, H, W, znear=0.001, zfar=1000):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]
    P = torch.zeros(4, 4, dtype=K.dtype, device=K.device)
    z_sign = 1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear) # z_sign * 2 * zfar * znear / (zfar - znear)
    P[3, 2] = z_sign

    return P

def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y


class Camera:
    def __init__(self, w2c, intrinsic, FoVx, FoVy, height, width, znear, zfar, trans=np.array([0.0, 0.0, 0.0]), scale=1.0) -> None:
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = height
        self.width = width
        self.world_view_transform = w2c.transpose(0, 1)


        self.zfar = 1000.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.projection_matrix = getProjectionMatrix_refine(intrinsic, self.height, self.width, self.znear, self.zfar).transpose(0, 1).to(w2c.device)
        
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def from_w2c(w2c, intrinsic, height, width, znear, zfar):
        FoVx, FoVy = intrinsic_to_fov(intrinsic, w=torch.tensor(width, device=w2c.device), h=torch.tensor(height, device=w2c.device))
        return Camera(w2c=w2c, intrinsic=intrinsic, FoVx=FoVx, FoVy=FoVy, height=height, width=width, znear=znear, zfar=zfar)

class GaussianModel(NamedTuple):
    xyz: Tensor
    opacity: Tensor
    rotation: Tensor
    scaling: Tensor
    shs: Tensor
    density: Tensor
    frequencies: Optional[Tensor] = None
    phases: Optional[Tensor] = None
    amplitudes: Optional[Tensor] = None
    offsets: Optional[Tensor] = None
    omega: Optional[float] = None

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]
        for i in range(features_dc.shape[1]*features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_rest.shape[1]*features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        
        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]
        f_dc = features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(torch.clamp(self.opacity, 1e-3, 1 - 1e-3).detach().cpu().numpy())
        scale = np.log(self.scaling.detach().cpu().numpy())
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

class GSLayer(nn.Module):
    @dataclass
    class Config:
        in_channels: int = 128
        feature_channels: dict = field(default_factory=dict)
        xyz_offset: bool = True
        restrict_offset: bool = False
        use_rgb: bool = False
        clip_scaling: Optional[float] = None
        init_scaling: float = -5.0
        init_density: float = 0.1
        sh_degree: int = 2        # choose your SH degree
        n_neurons: int = 8        # number of MLP neurons
        spatial_lr_scale: float = 1.0  # learning‐rate scaling used when initializing from pcd

    cfg: Config
    gaussians: GaussianModelMLP_HB

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Important: register as submodule
        self.gaussians = GaussianModelMLP_HB(
            sh_degree=self.cfg.sh_degree,
            n_neurons=self.cfg.n_neurons
        )

        
        
    def initialize_from_point_cloud(self, pcd, training_args, toy_example=False):
        # Create the Gaussian representation from the point cloud
        self.gaussians.create_from_pcd(
            pcd=pcd,
            spatial_lr_scale=self.cfg.spatial_lr_scale,
            toy_example=toy_example
        )
        # Set up optimizer and gradients
        self.gaussians.training_setup(training_args)

    
    def initialize_from_gaussians(self, gs: GaussianModel, training_args, toy_example=False):
        """
        Deterministically initialise neural primitives using an existing classic GaussianModel.

        Copies center positions, SH features, scaling, rotation, opacity, and
        computes density for the neural GaussianModelMLP_HB instance.
        Frequencies / phases / amplitudes / offsets are left random.
        """

        # convert the classic GaussianModel to a point cloud for MLP initialization
        pcd = pcd_from_gaussians(gs)
        self.initialize_from_point_cloud(pcd, training_args, toy_example=toy_example)

        g = self.gaussians

        # Convert classic outputs into the internal parameter domain expected by
        # GaussianModelMLP_HB before copying.
        eps = 1e-6
        with torch.no_grad():
            g._xyz.copy_(gs.xyz.detach())
            g._rotation.copy_(gs.rotation.detach())

            # Store the initialization anchor points so we can apply learned
            # offsets to novel-pose points at inference/training time.
            g._xyz_init = gs.xyz.detach().clone()
            g._use_pose_residual = True

            # Classic path stores scaling in linear space; neural model stores log-scale.
            scaling_lin = torch.clamp(gs.scaling.detach(), min=eps)
            g._scaling.copy_(torch.log(scaling_lin))

            # Classic path stores opacity after sigmoid; neural model stores logits.
            alpha = torch.clamp(gs.opacity.detach(), min=eps, max=1.0 - eps)
            opacity_logit = torch.logit(alpha)
            g._opacity.copy_(opacity_logit)
            g._opacity_volr.copy_(g.inverse_opacity_volr_activation(alpha))

        # copy SH features (color)
        if hasattr(gs, 'shs'):
            shs = gs.shs.detach()  # [N, n_sh, 3]
            # target SH sizes in the neural model
            n_sh_target = g._features_dc.shape[1] + g._features_rest.shape[1]

            # pad or truncate if needed
            if shs.shape[1] != n_sh_target:
                if shs.shape[1] > n_sh_target:
                    shs = shs[:, :n_sh_target, :]
                else:
                    pad_amount = n_sh_target - shs.shape[1]
                    pad = shs.new_zeros(shs.shape[0], pad_amount, shs.shape[2])
                    shs = torch.cat([shs, pad], dim=1)

            # copy DC and rest SH coefficients
            g._features_dc.data.copy_(shs[:, :1, :])
            if g._features_rest.numel() > 0:
                g._features_rest.data.copy_(shs[:, 1:, :])

        # print(f"[initialize_from_gaussians] copied SHs: "
        #     f"dc shape={g._features_dc.shape}, dc_mean={g._features_dc.mean().item():.6f}, "
        #     f"rest_mean={g._features_rest.mean().item():.6f}")
        # print(f"[initialize_from_gaussians] opacity and density initialized")

    def get_gaussians(self):
        return self.gaussians
    
    def forward(self, x, pts):
        # neural version does not compute gaussians from x; they are stored inside
        if self.gaussians is None:
            raise ValueError("GSLayer gaussians not initialized. Call initialize_from_point_cloud() first.")
        gs = self.gaussians
        xyz_abs = gs.get_xyz

        # Classic-style propagation: current posed points plus learned residual.
        use_pose_residual = (
            hasattr(gs, "_xyz_init")
            and gs._xyz_init.shape == xyz_abs.shape
            and pts.shape[0] == xyz_abs.shape[0]
        )
        if use_pose_residual:
            xyz = pts + (xyz_abs - gs._xyz_init)
        elif pts.shape[0] == xyz_abs.shape[0]:
            xyz = pts
        else:
            xyz = xyz_abs
        
        opacity = gs.get_opacity
        density = gs.get_density
        scaling = gs.get_scaling
        features = gs.get_features
        rotation = gs.get_rotation
        frequencies = gs.get_frequencies
        phases = gs.get_phases
        amplitudes = gs.get_amplitudes
        offsets = gs.get_offsets
        omega = gs.omega
        # reshape features into SH coefficients of size 3
        shs = features.view(features.shape[0], -1, 3)

        return GaussianModel(xyz=xyz, opacity=opacity, rotation=rotation,
                     scaling=scaling, shs=shs, density=density,
                     frequencies=frequencies, phases=phases,
                     amplitudes=amplitudes, offsets=offsets,
                     omega=omega)



class GSLayer_non_neural(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 128
        feature_channels: dict = field(default_factory=dict)
        xyz_offset: bool = True
        restrict_offset: bool = False
        use_rgb: bool = False
        clip_scaling: Optional[float] = None
        init_scaling: float = -5.0
        init_density: float = 0.1
        n_neurons: int = 8        # number of MLP neurons
        sh_degree: int = 2        # choose your SH degree
        spatial_lr_scale: float = 1.0  # learning‐rate scaling used when initializing from pcd
        
    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        self.out_layers = nn.ModuleList()
        for key, out_ch in self.cfg.feature_channels.items():
            if key == "shs" and self.cfg.use_rgb:
                out_ch = 3
            layer = nn.Linear(self.cfg.in_channels, out_ch)

            # initialize
            if not (key == "shs" and self.cfg.use_rgb):
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, self.cfg.init_scaling)
            elif key == "rotation":
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                nn.init.constant_(layer.bias, inverse_sigmoid(self.cfg.init_density))
            elif key == "density":
                nn.init.constant_(layer.bias, inverse_sigmoid(self.cfg.init_density))
            self.out_layers.append(layer)

    def forward(self, x, pts):
        ret = {}
        for k, layer in zip(self.cfg.feature_channels.keys(), self.out_layers):
            v = layer(x)
            if k == "rotation":
                v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = trunc_exp(v)
                if self.cfg.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.cfg.clip_scaling)
            elif k == "opacity":
                v = torch.sigmoid(v)
            elif k == "shs":
                if self.cfg.use_rgb:
                    v = torch.sigmoid(v)
                v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                if self.cfg.restrict_offset:
                    max_step = 1.2 / 32
                    v = (torch.sigmoid(v) - 0.5) * max_step
                v = v + pts if self.cfg.xyz_offset else pts
            elif k == "density":
                alpha = torch.sigmoid(v)  # original per-primitive opacity
                # convert to neural density using Splat-the-Net formula
                v = alpha2density(alpha, ret["scaling"] , reparam_type="ours")
            ret[k] = v

        return GaussianModel(**ret)
    
class GS3DRenderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        mlp_network_config: Optional[dict] = None
        gs_out: dict = field(default_factory=dict)
        sh_degree: int = 3
        render_backend: str = "splatnet_mlpmax"  # choices: "classic", "splatnet_mlpmax"
        compute_cov3D_python: bool = False
        scaling_modifier: float = 1.0
        random_background: bool = False
        radius: float = 1.0
        radius_texture: float = 1.0
        feature_reduction: str = "concat"
        projection_feature_dim: int = 773
        background_color: Tuple[float, float, float] = field(
            default_factory=lambda: (1.0, 1.0, 1.0)
        )

    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        if self.cfg.feature_reduction == "mean":
            mlp_in = 80
        elif self.cfg.feature_reduction == "concat":
            mlp_in = 80 * 3
        else:
            raise NotImplementedError
        mlp_in = 80+50+1
        if self.cfg.mlp_network_config is not None:
            self.mlp_net = MLP(mlp_in, self.cfg.gs_out.in_channels, **self.cfg.mlp_network_config)
        else:
            self.cfg.gs_out.in_channels = mlp_in
        # make gaussian layer use same SH degree as rasterizer
        self.cfg.gs_out.sh_degree = self.cfg.sh_degree
        self.gs_net= GSLayer(self.cfg.gs_out)
        self.gs_valid = vert_valid(verts_f_dim = mlp_in)
        # self.inter_attn = inter_attn(f_dim = mlp_in)
        self.self_attn_layer = SelfAttn(f_dim = mlp_in)
        self.vert_pos_refinement = vert_pos_refinement(verts_f_dim = mlp_in)
        self.threshold_low = 0.1
        self.threshold_high = 0.9
        if self.cfg.render_backend == "splatnet_mlpmax" and not HAS_MLP_MAX_RASTERIZER:
            raise ImportError(
                "render_backend='splatnet_mlpmax' requires diff_gaussian_mlp_max_rasterization. "
                "Build/install it from splat-the-net/submodules/diff-gaussian-mlp-max-rasterization."
            )

    def forward_gs(self, x, p):
        if self.cfg.mlp_network_config is not None:
            x = self.mlp_net(x)
        return self.gs_net(x, p)

    def forward_single_view(self,
        gs: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ret_mask: bool = True,
        color_w = None,
        xyz_b = None,
        color_b = None,
        opacity_b = None,
        ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        bg_color = background_color
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.cfg.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform.float(),
            sh_degree=self.cfg.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=False,
        )

        means3D = gs.xyz

        if xyz_b is not None:
            means3D = means3D + xyz_b

        means2D = screenspace_points
        opacity = gs.opacity

        if opacity_b != None:
            opacity = opacity + opacity_b.view(-1,1)

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.cfg.compute_cov3D_python and hasattr(self.gs_net.gaussians, "get_covariance"):
            cov3D_precomp = self.gs_net.gaussians.get_covariance(self.cfg.scaling_modifier)
        else:
            scales = gs.scaling
            rotations = gs.rotation

        # Match classic rendering path: when `use_rgb` is enabled, interpret
        # `gs.shs[:, 0, :]` as precomputed RGB colors.
        shs = None
        colors_precomp = None
        if self.gs_net.cfg.use_rgb:
            n_sh = (self.cfg.sh_degree + 1) ** 2
            colors_precomp = gs.shs[:, 0, :]
            if color_w is not None:
                cw = color_w.view(-1, n_sh, 3)
                colors_precomp = colors_precomp * cw[:, 0, :] + cw[:, 1, :] - 1
            if color_b is not None:
                colors_precomp = colors_precomp + color_b.view(-1, n_sh, 3)[:, 0, :]
        else:
            shs = gs.shs
            n_sh = shs.shape[1]
            if color_w is not None:
                shs = shs*color_w.view(-1,n_sh,3)
            if color_b is not None:
                shs = shs + color_b.view(-1,n_sh,3)
        
        if self.cfg.render_backend == "splatnet_mlpmax":
            if gs.frequencies is None or gs.phases is None or gs.amplitudes is None or gs.offsets is None:
                raise RuntimeError("Splat-the-Net backend requires neural MLP primitive parameters.")

            mlp_raster_settings = Gaussian_MLP_RasterizationSettings(
                image_height=int(viewpoint_camera.height),
                image_width=int(viewpoint_camera.width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_color,
                scale_modifier=self.cfg.scaling_modifier,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform.float(),
                sh_degree=self.cfg.sh_degree,
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                debug=False,
                antialiasing=False,
            )
            rasterizer = Gaussian_MLP_Rasterizer(raster_settings=mlp_raster_settings)

            omega = gs.omega if gs.omega is not None else 30.0
            frequencies_boosted = gs.frequencies * omega
            phases_boosted = gs.phases * omega

            with torch.autocast(device_type=self.device.type, dtype=torch.float32):
                rendered_image, radii, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    shs=shs,
                    colors_precomp=colors_precomp,
                    opacities=gs.density,
                    scales=scales,
                    rotations=rotations,
                    weight1=frequencies_boosted,
                    bias1=phases_boosted,
                    weight2=gs.amplitudes,
                    bias2=gs.offsets,
                    cov3D_precomp=cov3D_precomp,
                )
        else:
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            with torch.autocast(device_type=self.device.type, dtype=torch.float32):
                rendered_image, radii, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = colors_precomp,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
        
        ret = {
            "comp_rgb": rendered_image.permute(1, 2, 0),
            "comp_rgb_bg": bg_color
        }
        
        if ret_mask:
            mask_bg_color = torch.zeros(3, dtype=torch.float32, device=self.device)
            if self.cfg.render_backend == "splatnet_mlpmax":
                mlp_mask_settings = Gaussian_MLP_RasterizationSettings(
                    image_height=int(viewpoint_camera.height),
                    image_width=int(viewpoint_camera.width),
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=mask_bg_color,
                    scale_modifier=self.cfg.scaling_modifier,
                    viewmatrix=viewpoint_camera.world_view_transform,
                    projmatrix=viewpoint_camera.full_proj_transform.float(),
                    sh_degree=0,
                    campos=viewpoint_camera.camera_center,
                    prefiltered=False,
                    debug=False,
                    antialiasing=False,
                )
                mlp_mask_rasterizer = Gaussian_MLP_Rasterizer(raster_settings=mlp_mask_settings)

                omega = gs.omega if gs.omega is not None else 30.0
                frequencies_boosted = gs.frequencies * omega
                phases_boosted = gs.phases * omega

                with torch.autocast(device_type=self.device.type, dtype=torch.float32):
                    rendered_mask, radii, _ = mlp_mask_rasterizer(
                        means3D=means3D,
                        means2D=means2D,
                        shs=None,
                        colors_precomp=torch.ones_like(means3D),
                        opacities=gs.density,
                        scales=scales,
                        rotations=rotations,
                        weight1=frequencies_boosted,
                        bias1=phases_boosted,
                        weight2=gs.amplitudes,
                        bias2=gs.offsets,
                        cov3D_precomp=cov3D_precomp,
                    )
                    ret["comp_mask"] = rendered_mask.permute(1, 2, 0)
            else:
                raster_settings = GaussianRasterizationSettings(
                    image_height=int(viewpoint_camera.height),
                    image_width=int(viewpoint_camera.width),
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=mask_bg_color,
                    scale_modifier=self.cfg.scaling_modifier,
                    viewmatrix=viewpoint_camera.world_view_transform,
                    projmatrix=viewpoint_camera.full_proj_transform.float(),
                    sh_degree=0,
                    campos=viewpoint_camera.camera_center,
                    prefiltered=False,
                    debug=False,
                    antialiasing=False
                )
                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                with torch.autocast(device_type=self.device.type, dtype=torch.float32):
                    rendered_mask, radii, _ = rasterizer(
                        means3D = means3D,
                        means2D = means2D,
                        colors_precomp = torch.ones_like(means3D),
                        opacities = opacity,
                        scales = scales,
                        rotations = rotations,
                        cov3D_precomp = cov3D_precomp)
                    ret["comp_mask"] = rendered_mask.permute(1, 2, 0)

        return ret
    
    def query_triplane(
        self,
        positions: Float[Tensor, "*B N 3"],
        triplanes: Float[Tensor, "*B 3 Cp Hp Wp"],
    ) -> Dict[str, Tensor]:
        batched = positions.ndim == 3
        if not batched:
            # no batch dimension
            triplanes = triplanes[None, ...]
            positions = positions[None, ...]
        positions=positions-positions.mean(-2).unsqueeze(-2)
        positions = scale_tensor(positions, (-self.cfg.radius, self.cfg.radius), (-1, 1))
        
        indices2D: Float[Tensor, "B 3 N 2"] = torch.stack(
                (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
                dim=-3,
            )
        out: Float[Tensor, "B3 Cp 1 N"] = F.grid_sample(
            rearrange(triplanes, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3),
            rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3),
            align_corners=True,
            mode="bilinear",
        )

        if self.cfg.feature_reduction == "concat":
            out = rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=3)
        elif self.cfg.feature_reduction == "mean":
            out = reduce(out, "(B Np) Cp () N -> B N Cp", Np=3, reduction="mean")
        else:
            raise NotImplementedError
        
        if not batched:
            out = out.squeeze(0)

        return out

    def query_triplane_texture(
        self,
        positions: Float[Tensor, "*B N 2"],
        triplanes: Float[Tensor, "*B 1 Cp Hp Wp"],
    ) -> Dict[str, Tensor]:
        batched = positions.ndim == 3
        if not batched:
            # no batch dimension
            triplanes = triplanes[None, ...]
            positions = positions[None, ...]

        positions = scale_tensor(positions, (-self.cfg.radius_texture, self.cfg.radius_texture), (-1, 1))
        positions = torch.clamp(positions, min=-1.0 + 1e-4, max=1.0 - 1e-4)
 
        indices2D: Float[Tensor, "B N 2"] = positions[:, :, None]

        out: Float[Tensor, "B3 Cp 1 N"] = F.grid_sample(
            triplanes.squeeze(1),
            indices2D,
            align_corners=True,
            mode="bilinear",
        )

        out = out.view(*out.shape[:2], -1).permute(0, 2, 1)
        if not batched:
            out = out.squeeze(0)

        return out

    def forward_single_batch(
        self,
        gs_hidden_features: Float[Tensor, "Np Cp"],
        query_points: Float[Tensor, "Np 3"],
        w2cs: Float[Tensor, "Nv 4 4"],
        intrinsics: Float[Tensor, "Nv 4 4"],
        height: int,
        width: int,
        znear,
        zfar,
        background_color: Optional[Float[Tensor, "3"]],
        color_w = None,
        xyz_b = None,
        color_b = None,
        opacity_b = None,
        vert3d_uv = None, 
        face_uv=None, 
        face_uv_xy=None,
        render_edit = None,
    ):
       
        if_gs_valid = self.gs_valid(gs_hidden_features, query_points).squeeze(1)
        low_idx = torch.nonzero(if_gs_valid > self.threshold_low, as_tuple=False).squeeze(1)
        high_idx = torch.nonzero(if_gs_valid > self.threshold_high, as_tuple=False).squeeze(1)

        if low_idx.numel() == 0:
            low_idx = torch.arange(query_points.shape[0], device=query_points.device)

        query_points_valid = query_points[low_idx]
        gs_hidden_features_valid = gs_hidden_features[low_idx]

        query_points_copied = query_points[high_idx]
        gs_hidden_features_copied = gs_hidden_features[high_idx]
        if query_points_copied.shape[0] > 0:
            query_points_copied = self.vert_pos_refinement(gs_hidden_features_copied, query_points_copied)
            query_points_valid = torch.cat([query_points_valid, query_points_copied], dim=-2)
            gs_hidden_features_copied = torch.cat([gs_hidden_features_valid, gs_hidden_features_copied], dim=-2)
        else:
            gs_hidden_features_copied = gs_hidden_features_valid
        # --- initialize neural primitives from old gaussians on first forward ---
        if not getattr(self.gs_net.gaussians, "_initialized", False):
            # create temporary non-neural layer with same output size
            train_args = make_default_training_args()
            device = gs_hidden_features_copied.device
            # GSLayer_non_neural.Config does not accept sh_degree, so strip it
            tmp_cfg = dict(self.cfg.gs_out)
            tmp_cfg.pop('sh_degree', None)
            tmp_layer = GSLayer_non_neural(tmp_cfg)
            tmp_layer.configure()
            tmp_layer = tmp_layer.to(device)   # move after configure so weights are on correct device
            # if an MLP is used in the real forward pass we must apply it
            tmp_x = gs_hidden_features_copied
            if self.cfg.mlp_network_config is not None:
                tmp_x = self.mlp_net(tmp_x)
            old_gs = tmp_layer(tmp_x, query_points_valid)
            # use deterministic initializer provided earlier
            self.gs_net.initialize_from_gaussians(old_gs, train_args)
            self.gs_net.gaussians._initialized = True
            print("Initialising neural primitives from existing GaussianModel, this should only happen once.")
            # ensure the newly created gaussian module lives on the right device

        gs: GaussianModel = self.forward_gs(gs_hidden_features_copied, query_points_valid)
        out_list = []
       
        # Sample color modulation at actual gaussian centers so sampled rows
        # always match gaussian count exactly.
        uv_query_points = gs.xyz
        vert_uv, vert_d, intermediates_vert = get_uvd(uv_query_points, vert3d_uv[0], face_uv, face_uv_xy)
        vert_uv = vert_uv.unsqueeze(0)
        vert_d = vert_d.unsqueeze(0)
        # normalize it to [-1, 1]
        vert_uv[..., 0] = 2.0 * (vert_uv[..., 0] /1) - 1.0
        vert_uv[..., 1] = 2.0 * (vert_uv[..., 1] /0.5) - 1.0

        # number of spherical harmonic channels used by the current renderer
        n_sh = (self.cfg.sh_degree + 1) ** 2
        color_w0 = torch.ones(size=(n_sh, 3, 1024, 2048))
        # old color_w parameter is flat (batch dim 1) so reshape first
        cw = color_w.view(-1, n_sh, 3)
        color_w0[0, ..., :1024] = cw[0, 0, :].unsqueeze(-1).unsqueeze(-1).repeat(1, 1024, 1024)
        color_w0[1, ..., :1024] = cw[0, 1, :].unsqueeze(-1).unsqueeze(-1).repeat(1, 1024, 1024)
        color_w0[0, ..., 1024:] = cw[0, 2, :].unsqueeze(-1).unsqueeze(-1).repeat(1, 1024, 1024)
        color_w0[1, ..., 1024:] = cw[0, 3, :].unsqueeze(-1).unsqueeze(-1).repeat(1, 1024, 1024)

        if render_edit != None:
            if render_edit['duplication']:
                color_w0[0,...,:1024], color_w0[1,...,:1024] = color_w0[0,...,1024:], color_w0[1,...,1024:]

        color_w0 = color_w0.view(-1,1024,2048)
        color_w = self.query_triplane_texture(vert_uv, color_w0.to(vert_uv.device).unsqueeze(0).unsqueeze(0)).squeeze(0)

        # after sampling we may have one weight per query-point: later
        # the gaussian model may contain a different number of primitives if
        # thresholds changed.  Trim (or pad) so colour arrays align with
        # the tensor returned by `gs`.
        if gs is not None:
            n_gauss = gs.xyz.shape[0]
            if color_w.shape[0] != n_gauss:
                if color_w.shape[0] > n_gauss:
                    print(f"[warning] sampled color_w for {color_w.shape[0]} points but gaussians={n_gauss}, trimming")
                    color_w = color_w[:n_gauss]
                else:
                    # pad with ones (identity) for missing points
                    pad = color_w.new_ones(n_gauss - color_w.shape[0], color_w.shape[1])
                    print(f"[warning] sampled color_w for {color_w.shape[0]} points but gaussians={n_gauss}, padding {pad.shape[0]} rows")
                    color_w = torch.cat([color_w, pad], dim=0)
        if color_b != None:
            if render_edit != None:
                if render_edit['edit_left_only']:
                    color_b[...,:1024] = 0
                if render_edit['duplication']:
                    color_b = torch.cat([color_b[...,1024:],color_b[...,1024:]],dim=-1)
            color_b = self.query_triplane_texture(vert_uv, color_b.unsqueeze(0).unsqueeze(0)).squeeze(0)
            if gs is not None:
                n_gauss = gs.xyz.shape[0]
                if color_b.shape[0] != n_gauss:
                    if color_b.shape[0] > n_gauss:
                        print(f"[warning] sampled color_b for {color_b.shape[0]} points but gaussians={n_gauss}, trimming")
                        color_b = color_b[:n_gauss]
                    else:
                        pad = color_b.new_zeros(n_gauss - color_b.shape[0], *color_b.shape[1:])
                        print(f"[warning] sampled color_b for {color_b.shape[0]} points but gaussians={n_gauss}, padding {pad.shape[0]} rows")
                        color_b = torch.cat([color_b, pad], dim=0)
        if opacity_b != None:
            opacity_b = self.query_triplane_texture(vert_uv, opacity_b.unsqueeze(0).unsqueeze(0)).squeeze(0)

        for w2c, intrinsic in zip(w2cs, intrinsics):
            out_list.append(self.forward_single_view(
                                gs, 
                                Camera.from_w2c(w2c = w2c, intrinsic = intrinsic, height = height, width = width, znear = znear, zfar = zfar),
                                background_color,
                                color_w = color_w,
                                xyz_b = xyz_b,
                                color_b = color_b,
                                opacity_b = opacity_b,
                            ))
        
        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        out["3dgs"] = gs
        return out

    def forward(self, 
        scene_codes_texture,
        vert_uv,
        query_points: Float[Tensor, "B Np 3"],
        w2c: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height,
        width,
        znear=0.71, 
        zfar=1.42,
        query_points_tar=None,
        additional_features: Optional[Float[Tensor, "B C H W"]] = None,
        background_color: Optional[Float[Tensor, "B 3"]] = None,
        intrinsic_input: Float[Tensor, "B Nv 4 4"] = None,
        w2c_input: Float[Tensor, "B Nv 4 4"] = None,
        texture_rgb = None,
        face = None,
        gs_hidden_features: Float[Tensor, "B Np Cp"] = None,
        mink_idxs_inter = None,
        color_w = None,
        xyz_b = None,
        color_b = None,
        opacity_b = None,
        vert3d_uv=None, 
        face_uv=None, 
        face_uv_xy=None,
        aux_cam_w2c=None,
        aux_cam_intrinsic=None,
        render_edit = None,
        **kwargs):
        batch_size = scene_codes_texture.shape[0]

        out_list = []
        out_list_input = []

        gs_hidden_features_texture = self.query_triplane_texture(vert_uv, scene_codes_texture)
        gs_hidden_features = gs_hidden_features_texture

        if additional_features is not None:
            gs_hidden_features = torch.cat([gs_hidden_features, additional_features], dim=-1)

        mink_idxs_inter = mink_idxs_inter.squeeze(-1)
        if mink_idxs_inter.max()>0:
            gs_hidden_features_list = []
            for b in range(batch_size):
                gs_hidden_features_b = gs_hidden_features[b].unsqueeze(0)
                mink_idxs_inter_b = mink_idxs_inter[b].unsqueeze(0)
                if mink_idxs_inter_b.max()>0:
                    if mink_idxs_inter.shape[1] >30000:
                        n_part = 8
                        part_len = int(mink_idxs_inter.shape[1]/n_part)
                        for p in range(n_part):
                            mink_idxs_inter_b_part = mink_idxs_inter_b*0
                            mink_idxs_inter_b_part[:,part_len*p:part_len*(p+1)] = mink_idxs_inter_b[:,part_len*p:part_len*(p+1)]
                            mink_idxs_inter_b_part = mink_idxs_inter_b_part.bool()
                            if mink_idxs_inter_b_part.max()<0.5:
                                continue
                            gs_hidden_features_b[mink_idxs_inter_b_part] = self.self_attn_layer(gs_hidden_features_b[mink_idxs_inter_b_part].unsqueeze(0)).squeeze(0)
                    else:
                        gs_hidden_features_b[mink_idxs_inter_b] = self.self_attn_layer(gs_hidden_features_b[mink_idxs_inter_b].unsqueeze(0)).squeeze(0)
                gs_hidden_features_list.append(gs_hidden_features_b)
            gs_hidden_features = torch.cat(gs_hidden_features_list, dim=0)

        if query_points_tar is not None:
            query_points = query_points_tar

        out = defaultdict(list)
        out_input = defaultdict(list)

        for b in range(batch_size):
            out_list_input.append(self.forward_single_batch(
                gs_hidden_features=gs_hidden_features[b],
                query_points=query_points[b],
                w2cs=w2c_input[b],
                intrinsics=intrinsic_input[b],
                height=height, 
                width=width,
                znear=znear,
                zfar=zfar,
                background_color=background_color[b] if background_color is not None else None,
                color_w = color_w,
                xyz_b = xyz_b,
                color_b = color_b,
                opacity_b = opacity_b,
                vert3d_uv=vert3d_uv, 
                face_uv=face_uv, 
                face_uv_xy=face_uv_xy,
                render_edit = render_edit,
                ),
            )
            
        for out_ in out_list_input:
            for k, v in out_.items():
                out_input[k+'_input'].append(v)
        for k, v in out_input.items():
            if isinstance(v[0], torch.Tensor):
                out_input[k] = torch.stack(v, dim=0)
            else:
                out_input[k] = v

        for k, v in out_input.items():
            out[k] = v

        return out
        
