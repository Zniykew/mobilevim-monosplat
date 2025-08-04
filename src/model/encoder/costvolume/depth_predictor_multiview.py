import torch
torch.autograd.set_detect_anomaly(True)

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..backbone.unimatch.geometry import coords_grid
from .ldm_unet.unet import UNetModel
from .dpt import DPTHead, CostHead
from .mv_transformer import (
    MultiViewFeatureTransformer,
)
from .utils import mv_feature_add_position
from ..mobilevim2 import mobilevim_xxs


def warp_with_pose_depth_candidates(
        feature1,
        intrinsics,
        pose,
        depth,
        clamp_min_depth=1e-3,
        warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    # # 添加调试信息
    # print(f"[warp_with_pose_depth_candidates] feature1 shape: {feature1.shape}")
    # print(f"[warp_with_pose_depth_candidates] depth shape: {depth.shape}")

    with torch.no_grad():
        # pixel coordinates - 对coords_grid的结果进行clone
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        ).clone()  # [B, 3, H, W] - 添加.clone()

        # back project to 3D and transform viewpoint
        # 确保所有操作都不是inplace操作
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1)).clone()  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ).clone() * depth.view(
            b, 1, d, h * w
        ).clone()  # [B, 3, D, H*W]

        points = (points + pose[:, :3, -1:].unsqueeze(-1)).clone()  # [B, 3, D, H*W]

        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        ).clone()  # [B, 3, D, H*W]

        # 确保所有操作都不是inplace操作
        pixel_coords = (points[:, :2].clone() / points[:, -1:].clone().clamp(
            min=clamp_min_depth
        )).clone()  # [B, 2, D, H*W]

        # normalize to [-1, 1] - 确保所有操作都不是inplace操作
        x_grid = (2 * pixel_coords[:, 0].clone() / (w - 1) - 1).clone()
        y_grid = (2 * pixel_coords[:, 1].clone() / (h - 1) - 1).clone()

        grid = torch.stack([x_grid, y_grid], dim=-1).clone()  # [B, D, H*W, 2]

    # sample features
    # 确保feature1不被inplace修改，总是clone
    feature1 = feature1.clone()

    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c, d, h, w
    )  # [B, C, D, H, W]

    return warped_feature



def prepare_feat_proj_data_lists(features, intrinsics, extrinsics, num_reference_views, idx):
    b, v, c, h, w = features.shape
    idx = idx[:, :, 1:]  # remove the current view
    if extrinsics is not None:
        # extract warp poses
        # 确保所有操作都不是inplace操作
        idx_to_warp = repeat(idx, "b v m -> b v m fw fh", fw=4, fh=4).clone()  # [b, v, m, 1, 1]
        extrinsics_cur = repeat(extrinsics.clone().detach(), "b v fh fw -> b v m fh fw",
                                m=num_reference_views).clone()  # [b, v, 4, 4]
        poses_others = extrinsics_cur.gather(1, idx_to_warp).clone()  # [b, v, m, 4, 4]
        poses_others_inv = torch.linalg.inv(poses_others).clone()  # [b, v, m, 4, 4]
        poses_cur = extrinsics.clone().detach().unsqueeze(2).clone()  # [b, v, 1, 4, 4]
        poses_warp = (poses_others_inv @ poses_cur).clone()  # [b, v, m, 4, 4]
        poses_warp = rearrange(poses_warp, "b v m ... -> (b v) m ...").clone()  # [bxv, m, 4, 4]
    else:
        poses_warp = None

    if features is not None:
        # extract warp features
        # 关键修改：确保所有操作都不是inplace操作
        idx_to_warp = repeat(idx, "b v m -> b v m c h w", c=c, h=h, w=w).clone()  # [b, v, m, 1]
        features_cur = repeat(features, "b v c h w -> b v m c h w", m=num_reference_views).clone()  # [b, v, m, c, h, w]
        feat_warp = features_cur.gather(1, idx_to_warp).clone()  # [b, v, m, c, h, w]
        feat_warp = rearrange(feat_warp, "b v m c h w -> (b v) m c h w").clone()  # [bxv, m, c, h, w]
    else:
        feat_warp = None

    if intrinsics is not None:
        # extract warp intrinsics
        # 关键修改：确保所有操作都不是inplace操作
        intr_curr = intrinsics[:, :, :3, :3].clone().detach().clone()  # [b, v, 3, 3]
        intr_curr = (intr_curr.clone() * 1).clone()  # 避免inplace乘法
        intr_curr[:, :, 0, :] = (intr_curr[:, :, 0, :].clone() * float(w)).clone()
        intr_curr[:, :, 1, :] = (intr_curr[:, :, 1, :].clone() * float(h)).clone()
        idx_to_warp = repeat(idx, "b v m -> b v m fh fw", fh=3, fw=3).clone()  # [b, v, m, 1, 1]
        intr_curr = repeat(intr_curr, "b v fh fw -> b v m fh fw", m=num_reference_views).clone()  # [b, v, m, 3, 3]
        intr_warp = intr_curr.gather(1, idx_to_warp).clone()  # [b, v, m, 3, 3]
        intr_warp = rearrange(intr_warp, "b v m ... -> (b v) m ...").clone()  # [bxv, m, 3, 3]
    else:
        intr_warp = None

    return feat_warp, intr_warp, poses_warp



class DepthPredictorMultiView(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
            self,
            feature_channels=128,
            upscale_factor=4,
            num_depth_candidates=32,
            costvolume_unet_feat_dim=128,
            costvolume_unet_channel_mult=(1, 1, 1),
            costvolume_unet_attn_res=(),
            gaussian_raw_channels=-1,
            gaussians_per_pixel=1,
            num_views=2,
            depth_unet_feat_dim=64,
            depth_unet_attn_res=(),
            depth_unet_channel_mult=(1, 1, 1),
            num_transformer_layers=3,
            num_head=1,
            ffn_dim_expansion=4,
            **kwargs,
    ):
        super(DepthPredictorMultiView, self).__init__()
        self.num_depth_candidates = num_depth_candidates
        self.regressor_feat_dim = costvolume_unet_feat_dim
        self.upscale_factor = upscale_factor
        self.feature_channels = feature_channels

        # 使用 MobileViM 替换 DINOv2
        self.mobilevim = mobilevim_xxs(num_classes=1000)
        # 冻结 MobileViM 参数
        for param in self.mobilevim.parameters():
            param.requires_grad = False

        # MobileViM 的嵌入维度（根据 MobileViM 配置调整）
        self.mobilevim_embed_dims = [192, 384, 448]

        # 特征投影层，将 MobileViM 特征映射到 DINOv2 相同维度 (384)
        self.feature_projections = nn.ModuleList([
            nn.Linear(embed_dim, 384) for embed_dim in [192, 384, 448, 512]  # 添加第4层
        ])

        # 适配器层，将 MobileViM 特征转换为 DINOv2 格式
        self.depth_head = DPTHead(384,
                                  features=feature_channels,
                                  use_bn=False,
                                  out_channels=[48, 96, 192, 384],
                                  use_clstoken=False)
        for param in self.depth_head.parameters():
            param.requires_grad = False

        self.cost_head = CostHead(384,
                                  features=feature_channels,
                                  use_bn=False,
                                  out_channels=[48, 96, 192, 384],
                                  use_clstoken=False)

        # Transformer
        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )

        # Cost volume refinement
        input_channels = num_depth_candidates + feature_channels * 2
        channels = self.regressor_feat_dim
        self.corr_refine_net = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=costvolume_unet_attn_res,
                channel_mult=costvolume_unet_channel_mult,
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=num_views,
                use_cross_view_self_attn=True,
            ),
            nn.Conv2d(channels, num_depth_candidates, 3, 1, 1))
        # cost volume u-net skip connection
        self.regressor_residual = nn.Conv2d(input_channels, num_depth_candidates, 1, 1, 0)

        # depth head
        self.depth_head_lowres = nn.Sequential(
            nn.Conv2d(num_depth_candidates, num_depth_candidates * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_depth_candidates * 2, num_depth_candidates, 3, 1, 1),
        )
        self.to_disparity = nn.Sequential(
            nn.Conv2d(
                feature_channels + 2 * gaussians_per_pixel + 1, feature_channels, 3, 1, 1
            ),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(feature_channels, gaussians_per_pixel * 2, 3, 1, 1),
        )
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(
                feature_channels * 3 + 3,  # 64*3 + 3 = 195
                feature_channels * 2,  # 128
                3,
                1,
                1,
            ),
            nn.GELU(),
            nn.Conv2d(feature_channels * 2, feature_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(feature_channels, gaussian_raw_channels, 3, 1, 1),
        )
        # feature projection
        self.proj_feature_mv = nn.Conv2d(feature_channels, feature_channels, 1, 1, 0)
        self.proj_feature_mono = nn.Conv2d(feature_channels, feature_channels, 1, 1, 0)

        # image feature extraction
        self.refine_unet = UNetModel(
            image_size=None,
            in_channels=feature_channels * 2 + 3 + gaussians_per_pixel * 2 + 1,
            model_channels=depth_unet_feat_dim,
            out_channels=feature_channels,
            num_res_blocks=2,
            attention_resolutions=depth_unet_attn_res,
            channel_mult=depth_unet_channel_mult,
            num_head_channels=32,
            dims=2,
            postnorm=True,
            num_frames=num_views,
            use_cross_view_self_attn=False,
        )

    def normalize_images(self, images):
        """Normalize image to match the pretrained UniMatch model.
        images: (B, V, C, H, W)
        """
        # 关键修改：确保输入images被clone
        images = images.clone()
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        # 确保所有操作都不是inplace操作
        return ((images - mean).clone() / std).clone()

    def extract_mobilevim_features(self, images):
        """使用 MobileViM 提取特征，替代 DINOv2，并确保输出格式与DINOv2一致"""
        b, c, h, w = images.shape

        # MobileViM 前向传播
        with torch.no_grad():
            x = self.mobilevim.patch_embed(images)

            features = []
            for stage in self.mobilevim.stages:
                for block in stage:
                    x = block(x)
                features.append(x)

            # 取最后4层特征（MobileViM 有4个阶段）
            selected_features = features[-4:] if len(features) >= 4 else features

        # 调整特征形状以匹配 DINOv2 的输出格式 [B, 324, 384]
        processed_features = []

        for i, feat in enumerate(selected_features):
            B, C, H, W = feat.shape
            # 将特征重塑为 [B, H*W, C] 格式
            feat_reshaped = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            # 投影到 DINOv2 的维度 (384)
            projected_feat = self.feature_projections[i](feat_reshaped)  # [B, H*W, 384]

            # 确保输出形状为 [B, 324, 384]
            if projected_feat.shape[1] != 324:
                # 通过插值或裁剪调整到324个位置
                current_h = int(projected_feat.shape[1] ** 0.5)
                current_w = projected_feat.shape[1] // current_h
                feat_2d = projected_feat.transpose(1, 2).reshape(B, -1, current_h, current_w)
                # 插值到18x18 (324个位置)
                feat_2d = F.interpolate(feat_2d, size=(18, 18), mode='bilinear', align_corners=False)
                projected_feat = feat_2d.flatten(2).transpose(1, 2)

            processed_features.append((projected_feat, 18, 18))  # 18*18=324

        return processed_features

    def _ensure_feature_shape(self, feature, target_length=324):
        """确保特征具有指定的长度"""
        B, N, C = feature.shape
        if N == target_length:
            return feature
        elif N > target_length:
            # 裁剪特征
            return feature[:, :target_length, :]
        else:
            # 通过插值扩展特征
            # 先reshape到2D网格
            h = int(N ** 0.5)
            w = N // h
            while h * w != N:
                h -= 1
                w = N // h

            feature_2d = feature.transpose(1, 2).reshape(B, C, h, w)

            # 计算目标尺寸
            target_h = int(target_length ** 0.5)
            target_w = target_length // target_h
            while target_h * target_w != target_length:
                target_h -= 1
                target_w = target_length // target_h

            # 插值到目标尺寸
            feature_2d = F.interpolate(feature_2d, size=(target_h, target_w), mode='bilinear', align_corners=False)
            return feature_2d.flatten(2).transpose(1, 2)

    def forward(
            self,
            images,
            intrinsics,
            extrinsics,
            near,
            far,
            gaussians_per_pixel=1,
            deterministic=True,
    ):
        # print(f"[DepthPredictorMultiView] === FORWARD START ===")
        # print(f"[DepthPredictorMultiView] Input shapes:")
        # print(f"[DepthPredictorMultiView]   images: {images.shape}")
        # print(f"[DepthPredictorMultiView]   intrinsics: {intrinsics.shape}")
        # print(f"[DepthPredictorMultiView]   extrinsics: {extrinsics.shape}")
        # print(f"[DepthPredictorMultiView]   near: {near.shape}")
        # print(f"[DepthPredictorMultiView]   far: {far.shape}")
        # print(f"[DepthPredictorMultiView]   gaussians_per_pixel: {gaussians_per_pixel}")

        # 彻底克隆所有输入张量
        images = images.clone()
        intrinsics = intrinsics.clone()
        extrinsics = extrinsics.clone()
        near = near.clone()
        far = far.clone()

        num_reference_views = 1
        # find nearest idxs
        cam_origins = extrinsics[:, :, :3, -1].clone()  # [b, v, 3]
        distance_matrix = torch.cdist(cam_origins, cam_origins, p=2).clone()  # [b, v, v]
        _, idx = torch.topk(distance_matrix, num_reference_views + 1, largest=False, dim=2)  # [b, v, m+1]
        # print(f"[DepthPredictorMultiView] Camera distance matrix shape: {distance_matrix.shape}")
        # print(f"[DepthPredictorMultiView] Nearest neighbor indices shape: {idx.shape}")

        # first normalize images
        images = self.normalize_images(images)
        b, v, _, ori_h, ori_w = images.shape
        # print(f"[DepthPredictorMultiView] After normalization - images shape: {images.shape}")

        # depth anything encoder
        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14
        # print(f"[DepthPredictorMultiView] Resize dimensions: {resize_h} x {resize_w}")

        # 关键修改：使用独立的变量名避免任何可能的混淆
        concat_images = rearrange(images, "b v c h w -> (b v) c h w").clone()
        resized_images = F.interpolate(concat_images, (resize_h, resize_w), mode="bilinear", align_corners=True).clone()
        # print(f"[DepthPredictorMultiView] Concatenated images shape: {resized_images.shape}")

        # 使用 MobileViM 提取特征替代 DINOv2
        mobilevim_features = self.extract_mobilevim_features(resized_images)
        # print(f"[DepthPredictorMultiView] MobileViM features extracted - number of layers: {len(mobilevim_features)}")
        # for i, (feat, h, w) in enumerate(mobilevim_features):
        #     print(f"[DepthPredictorMultiView]   Feature layer {i}: {feat.shape}, spatial dims: {h}x{w}")

        # 重新组织特征格式以匹配后续处理
        features = []
        for feat, h, w in mobilevim_features:
            # 重新组织为 (feature_map, class_token) 格式以兼容后续代码
            features.append((feat.clone(), None))  # MobileViM 没有 class token

        # new decoder
        # 使用固定尺寸18x18匹配DINOv2
        features_mono, disps_rel = self.depth_head(features, patch_h=18, patch_w=18)
        features_mv = self.cost_head(features, patch_h=18, patch_w=18)
        # print(f"[DepthPredictorMultiView] After decoders:")
        # print(f"[DepthPredictorMultiView]   features_mono shape: {features_mono.shape}")
        # print(f"[DepthPredictorMultiView]   disps_rel shape: {disps_rel.shape}")
        # print(f"[DepthPredictorMultiView]   features_mv shape: {features_mv.shape}")

        # 确保中间特征形状与DINOv2版本一致
        # DINOv2版本输出的是 [2, 64, 144, 144]，我们需要保持一致
        if features_mono.shape[-2:] != (144, 144):
            features_mono = F.interpolate(features_mono, (144, 144), mode="bilinear", align_corners=True).clone()
        if features_mv.shape[-2:] != (144, 144):
            features_mv = F.interpolate(features_mv, (144, 144), mode="bilinear", align_corners=True).clone()
        if disps_rel.shape[-2:] != (252, 252):  # 保持原始尺寸而不是144x144
            disps_rel = F.interpolate(disps_rel, (252, 252), mode="bilinear", align_corners=True).clone()

        # print(f"[DepthPredictorMultiView] After interpolation to match DINOv2:")
        # print(f"[DepthPredictorMultiView]   features_mono shape: {features_mono.shape}")
        # print(f"[DepthPredictorMultiView]   disps_rel shape: {disps_rel.shape}")
        # print(f"[DepthPredictorMultiView]   features_mv shape: {features_mv.shape}")

        # 关键修改：彻底隔离所有张量
        features_mono = features_mono.clone().detach().requires_grad_(
            True) if features_mono.requires_grad else features_mono.clone()
        disps_rel = disps_rel.clone().detach().requires_grad_(True) if disps_rel.requires_grad else disps_rel.clone()
        features_mv = features_mv.clone().detach().requires_grad_(
            True) if features_mv.requires_grad else features_mv.clone()

        features_mv_upsampled = F.interpolate(features_mv, (64, 64), mode="bilinear", align_corners=True).clone()
        features_mv_pos = mv_feature_add_position(features_mv_upsampled, 2, 64).clone()
        features_mv_list = list(torch.unbind(rearrange(features_mv_pos, "(b v) c h w -> b v c h w", b=b, v=v), dim=1))
        # print(f"[DepthPredictorMultiView] After interpolation and position encoding:")
        # print(f"[DepthPredictorMultiView]   features_mv shape: {features_mv_pos.shape}")
        # print(f"[DepthPredictorMultiView]   features_mv_list length: {len(features_mv_list)}")
        # for i, feat in enumerate(features_mv_list):
        #     print(f"[DepthPredictorMultiView]   features_mv_list[{i}] shape: {feat.shape}")

        features_mv_list = self.transformer(
            features_mv_list,
            attn_num_splits=2,
            nn_matrix=idx.clone(),
        )

        # 关键修改：确保transformer输出也被隔离
        features_mv_transformed = rearrange(torch.stack(features_mv_list, dim=1), "b v c h w -> (b v) c h w")
        features_mv_transformed = features_mv_transformed.clone().detach().requires_grad_(
            True) if features_mv_transformed.requires_grad else features_mv_transformed.clone()
        # print(f"[DepthPredictorMultiView] After transformer:")
        # print(f"[DepthPredictorMultiView]   features_mv shape: {features_mv_transformed.shape}")

        # cost volume construction
        features_mv_reshaped = rearrange(features_mv_transformed, "(b v) c h w -> b v c h w", v=v, b=b).clone()
        features_mv_warped, intr_warped, poses_warped = (
            prepare_feat_proj_data_lists(
                features_mv_reshaped,
                intrinsics.clone(),
                extrinsics.clone(),
                num_reference_views=num_reference_views,
                idx=idx.clone()
            )
        )
        # print(f"[DepthPredictorMultiView] After prepare_feat_proj_data_lists:")
        # print(
        #     f"[DepthPredictorMultiView]   features_mv_warped shape: {features_mv_warped.shape if features_mv_warped is not None else None}")
        # print(
        #     f"[DepthPredictorMultiView]   intr_warped shape: {intr_warped.shape if intr_warped is not None else None}")
        # print(
        #     f"[DepthPredictorMultiView]   poses_warped shape: {poses_warped.shape if poses_warped is not None else None}")

        # 关键修改：构建视差候选
        far_clone = far.clone()
        near_clone = near.clone()
        min_disp = rearrange(1.0 / far_clone.detach(), "b v -> (b v) ()").clone()
        max_disp = rearrange(1.0 / near_clone.detach(), "b v -> (b v) ()").clone()
        disp_range_norm = torch.linspace(0.0, 1.0, self.num_depth_candidates, device=min_disp.device).clone()
        disp_candi_curr = (min_disp.unsqueeze(-1).unsqueeze(-1) +
                           disp_range_norm.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) *
                           (max_disp.unsqueeze(-1).unsqueeze(-1) - min_disp.unsqueeze(-1).unsqueeze(-1))).type_as(
            features_mv_transformed)
        disp_candi_curr = disp_candi_curr.clone()
        # print(f"[DepthPredictorMultiView] Disparity candidates shape: {disp_candi_curr.shape}")

        raw_correlation_in = []
        for i in range(num_reference_views):
            features_mv_warped_i = warp_with_pose_depth_candidates(
                features_mv_warped[:, i, :, :, :].clone(),
                intr_warped[:, i, :, :].clone(),
                poses_warped[:, i, :, :].clone(),
                1 / disp_candi_curr.clone(),
                warp_padding_mode="zeros"
            )
            # print(f"[DepthPredictorMultiView] After warp_with_pose_depth_candidates[{i}]: {features_mv_warped_i.shape}")

            # 关键修改：避免任何可能的就地操作
            correlation = torch.sum(
                features_mv_transformed.unsqueeze(2).clone() * features_mv_warped_i.clone(),
                dim=1
            ) / (features_mv_transformed.shape[1] ** 0.5)
            raw_correlation_in.append(correlation.clone())

        raw_correlation_in = torch.mean(torch.stack(raw_correlation_in, dim=1), dim=1).clone()
        # print(f"[DepthPredictorMultiView] Mean correlation shape: {raw_correlation_in.shape}")

        # refine cost volume and get depths
        features_mono_tmp = F.interpolate(features_mono, (64, 64), mode="bilinear", align_corners=True).clone()
        raw_correlation_in_combined = torch.cat((
            raw_correlation_in.clone(),
            features_mv_transformed.clone(),
            features_mono_tmp.clone()
        ), dim=1).clone()
        # print(f"[DepthPredictorMultiView] Before cost volume refinement:")
        # print(f"[DepthPredictorMultiView]   raw_correlation_in shape: {raw_correlation_in.shape}")

        raw_correlation = self.corr_refine_net(raw_correlation_in_combined).clone()
        raw_correlation_residual = self.regressor_residual(raw_correlation_in_combined.clone()).clone()
        raw_correlation = (raw_correlation + raw_correlation_residual).clone()
        # print(f"[DepthPredictorMultiView] After cost volume refinement:")
        # print(f"[DepthPredictorMultiView]   raw_correlation shape: {raw_correlation.shape}")

        pdf = F.softmax(self.depth_head_lowres(raw_correlation.clone()), dim=1).clone()
        disps_metric = torch.sum(disp_candi_curr * pdf, dim=1, keepdim=True).clone()
        pdf_max = torch.max(pdf, dim=1, keepdim=True)[0].clone()
        pdf_max = F.interpolate(pdf_max, (ori_h, ori_w), mode="bilinear", align_corners=True).clone()
        disps_metric_fullres = F.interpolate(disps_metric, (ori_h, ori_w), mode="bilinear", align_corners=True).clone()
        # print(f"[DepthPredictorMultiView] After depth estimation:")
        # print(f"[DepthPredictorMultiView]   pdf shape: {pdf.shape}")
        # print(f"[DepthPredictorMultiView]   disps_metric shape: {disps_metric.shape}")
        # print(f"[DepthPredictorMultiView]   pdf_max shape: {pdf_max.shape}")
        # print(f"[DepthPredictorMultiView]   disps_metric_fullres shape: {disps_metric_fullres.shape}")

        # feature refinement - 关键修改：避免任何就地操作
        features_mv_in_fullres = F.interpolate(features_mv_transformed, (ori_h, ori_w), mode="bilinear",
                                               align_corners=True).clone()
        features_mv_in_fullres = self.proj_feature_mv(features_mv_in_fullres.clone()).clone()

        features_mono_in_fullres = F.interpolate(features_mono, (ori_h, ori_w), mode="bilinear",
                                                 align_corners=True).clone()
        features_mono_in_fullres = self.proj_feature_mono(features_mono_in_fullres.clone()).clone()

        disps_rel_fullres = F.interpolate(disps_rel, (ori_h, ori_w), mode="bilinear", align_corners=True).clone()

        images_reorder = rearrange(images, "b v c h w -> (b v) c h w").clone()

        # print(f"[DepthPredictorMultiView] After feature interpolation:")
        # print(f"[DepthPredictorMultiView]   features_mv_in_fullres shape: {features_mv_in_fullres.shape}")
        # print(f"[DepthPredictorMultiView]   features_mono_in_fullres shape: {features_mono_in_fullres.shape}")
        # print(f"[DepthPredictorMultiView]   disps_rel_fullres shape: {disps_rel_fullres.shape}")
        # print(f"[DepthPredictorMultiView] images_reorder shape: {images_reorder.shape}")

        # 关键修改：构建UNet输入
        unet_input = torch.cat([
            features_mv_in_fullres.clone(),
            features_mono_in_fullres.clone(),
            images_reorder.clone(),
            disps_metric_fullres.clone(),
            disps_rel_fullres.clone(),
            pdf_max.clone()
        ], dim=1).clone()

        refine_out = self.refine_unet(unet_input).clone()

        # print(f"[DepthPredictorMultiView] After refinement UNet:")
        # print(f"[DepthPredictorMultiView]   refine_out shape: {refine_out.shape}")

        # gaussians head
        gaussians_input = torch.cat([
            refine_out.clone(),
            features_mv_in_fullres.clone(),
            features_mono_in_fullres.clone(),
            images_reorder.clone()
        ], dim=1).clone()
        raw_gaussians = self.to_gaussians(gaussians_input).clone()

        # print(f"[DepthPredictorMultiView] Gaussians prediction:")
        # print(f"[DepthPredictorMultiView]   raw_gaussians_in shape: {gaussians_input.shape}")
        # print(f"[DepthPredictorMultiView]   raw_gaussians shape: {raw_gaussians.shape}")

        # delta fine depth and density
        disparity_input = torch.cat([
            refine_out.clone(),
            disps_metric_fullres.clone(),
            disps_rel_fullres.clone(),
            pdf_max.clone()
        ], dim=1).clone()
        delta_disps_density = self.to_disparity(disparity_input).clone()
        delta_disps, raw_densities = delta_disps_density.split(gaussians_per_pixel, dim=1)

        # print(f"[DepthPredictorMultiView] Disparity and density prediction:")
        # print(f"[DepthPredictorMultiView]   disparity_in shape: {disparity_input.shape}")
        # print(f"[DepthPredictorMultiView]   delta_disps_density shape: {delta_disps_density.shape}")
        # print(f"[DepthPredictorMultiView]   delta_disps shape: {delta_disps.shape}")
        # print(f"[DepthPredictorMultiView]   raw_densities shape: {raw_densities.shape}")

        # outputs - 关键修改：完全避免就地操作
        far_rearranged = rearrange(far, "b v -> (b v) () () ()").clone()
        near_rearranged = rearrange(near, "b v -> (b v) () () ()").clone()

        # 使用torch.clamp而不是.clamp()方法
        fine_disps = torch.clamp(
            disps_metric_fullres.clone() + delta_disps.clone(),
            1.0 / far_rearranged.clone(),
            1.0 / near_rearranged.clone()
        ).clone()

        # 使用torch.div而不是除法操作符
        depths = torch.div(1.0, fine_disps.clone())
        depths = repeat(depths, "(b v) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1).clone()

        densities = repeat(
            torch.sigmoid(raw_densities.clone()),
            "(b v) dpt h w -> b v (h w) srf dpt",
            b=b,
            v=v,
            srf=1,
        ).clone()

        raw_gaussians = rearrange(raw_gaussians, "(b v) c h w -> b v (h w) c", v=v, b=b).clone()

        # print(f"[DepthPredictorMultiView] Final outputs:")
        # print(f"[DepthPredictorMultiView]   fine_disps shape: {fine_disps.shape}")
        # print(f"[DepthPredictorMultiView]   depths shape: {depths.shape}")
        # print(f"[DepthPredictorMultiView]   densities shape: {densities.shape}")
        # print(f"[DepthPredictorMultiView]   raw_gaussians shape: {raw_gaussians.shape}")
        # print(f"[DepthPredictorMultiView] === FORWARD END ===")

        return depths, densities, raw_gaussians