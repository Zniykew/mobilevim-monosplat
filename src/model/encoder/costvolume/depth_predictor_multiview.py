import torch
import copy

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

    with torch.no_grad():
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W]

        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]

        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]

        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]

        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
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
        idx_to_warp = repeat(idx, "b v m -> b v m fw fh", fw=4, fh=4)  # [b, v, m, 1, 1]
        extrinsics_cur = repeat(extrinsics, "b v fh fw -> b v m fh fw",
                                m=num_reference_views)  # [b, v, 4, 4]
        poses_others = extrinsics_cur.gather(1, idx_to_warp)  # [b, v, m, 4, 4]
        poses_others_inv = torch.linalg.inv(poses_others)  # [b, v, m, 4, 4]
        poses_cur = extrinsics.unsqueeze(2)  # [b, v, 1, 4, 4]
        poses_warp = torch.matmul(poses_others_inv, poses_cur)  # [b, v, m, 4, 4]
        poses_warp = rearrange(poses_warp, "b v m ... -> (b v) m ...")  # [bxv, m, 4, 4]
    else:
        poses_warp = None

    if features is not None:
        # extract warp features
        idx_to_warp = repeat(idx, "b v m -> b v m c h w", c=c, h=h, w=w)  # [b, v, m, 1]
        features_cur = repeat(features, "b v c h w -> b v m c h w", m=num_reference_views)  # [b, v, m, c, h, w]
        feat_warp = features_cur.gather(1, idx_to_warp)  # [b, v, m, c, h, w]
        feat_warp = rearrange(feat_warp, "b v m c h w -> (b v) m c h w")  # [bxv, m, c, h, w]
    else:
        feat_warp = None

    if intrinsics is not None:
        # extract warp intrinsics
        intr_curr = intrinsics[:, :, :3, :3].clone()  # [b, v, 3, 3] 添加 clone()
        intr_curr[:, :, 0, :] = intr_curr[:, :, 0, :] * float(w)
        intr_curr[:, :, 1, :] = intr_curr[:, :, 1, :] * float(h)
        idx_to_warp = repeat(idx, "b v m -> b v m fh fw", fh=3, fw=3)  # [b, v, m, 1, 1]
        intr_curr = repeat(intr_curr, "b v fh fw -> b v m fh fw", m=num_reference_views)  # [b, v, m, 3, 3]
        intr_warp = intr_curr.gather(1, idx_to_warp)  # [b, v, m, 3, 3]
        intr_warp = rearrange(intr_warp, "b v m ... -> (b v) m ...")  # [bxv, m, 3, 3]
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
        # MobileViM 参数
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
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return ((images - mean) / std)

    def extract_mobilevim_features(self, images):
        """使用 MobileViM 提取特征，替代 DINOv2"""
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
            # 将特征重塑为 [B, H*W, C] 格式，使用clone避免就地操作
            feat_reshaped = feat.flatten(2).transpose(1, 2).clone()  # [B, H*W, C]
            # 投影到 DINOv2 的维度 (384)
            projected_feat = self.feature_projections[i](feat_reshaped)  # [B, H*W, 384]

            # 确保输出形状为 [B, 324, 384]
            if projected_feat.shape[1] != 324:
                # 通过插值或裁剪调整到324个位置
                current_h = int(projected_feat.shape[1] ** 0.5)
                current_w = projected_feat.shape[1] // current_h
                feat_2d = projected_feat.transpose(1, 2).reshape(B, -1, current_h, current_w).clone()
                # 插值到18x18 (324个位置)
                feat_2d = F.interpolate(feat_2d, size=(18, 18), mode='bilinear', align_corners=False)
                projected_feat = feat_2d.flatten(2).transpose(1, 2).clone()

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
        num_reference_views = 1
        # find nearest idxs
        cam_origins = extrinsics[:, :, :3, -1]  # [b, v, 3]
        distance_matrix = torch.cdist(cam_origins, cam_origins, p=2)  # [b, v, v]
        _, idx = torch.topk(distance_matrix, num_reference_views + 1, largest=False, dim=2)  # [b, v, m+1]

        # first normalize images
        images = self.normalize_images(images)
        b, v, _, ori_h, ori_w = images.shape

        # depth anything encoder
        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14

        with torch.no_grad():
            concat_images = rearrange(images, "b v c h w -> (b v) c h w")
            resized_images = F.interpolate(concat_images, (resize_h, resize_w), mode="bilinear", align_corners=True)

        # 使用 MobileViM 提取特征替代 DINOv2
        mobilevim_features = self.extract_mobilevim_features(resized_images)

        # 重新组织特征格式以匹配后续处理，但保持梯度流
        features = []
        for feat, h, w in mobilevim_features:
            # 使用 clone 避免就地操作并保持梯度流
            features.append((feat.clone().detach(), None))

        # new decoder
        # 使用固定尺寸18x18匹配DINOv2
        features_mono, disps_rel = self.depth_head(features, patch_h=18, patch_w=18)
        features_mv = self.cost_head(features, patch_h=18, patch_w=18)

        # 确保中间特征形状与DINOv2版本一致，使用 clone 避免就地操作
        if features_mono.shape[-2:] != (144, 144):
            features_mono = F.interpolate(features_mono, (144, 144), mode="bilinear",
                                          align_corners=True).clone()
        if features_mv.shape[-2:] != (144, 144):
            features_mv = F.interpolate(features_mv, (144, 144), mode="bilinear", align_corners=True).clone()
        if disps_rel.shape[-2:] != (252, 252):
            disps_rel = F.interpolate(disps_rel, (252, 252), mode="bilinear", align_corners=True).clone()

        # 后续代码保持不变，但在关键位置添加 clone()
        features_mv_upsampled = F.interpolate(features_mv, (64, 64), mode="bilinear",
                                              align_corners=True).clone()
        features_mv_pos = mv_feature_add_position(features_mv_upsampled, 2, 64).clone()
        features_mv_list = list(torch.unbind(rearrange(features_mv_pos, "(b v) c h w -> b v c h w", b=b, v=v), dim=1))

        features_mv_list = self.transformer(
            features_mv_list,
            attn_num_splits=2,
            nn_matrix=idx.clone(),  # clone idx 以避免就地操作
        )

        features_mv_transformed = rearrange(torch.stack(features_mv_list, dim=1),
                                            "b v c h w -> (b v) c h w").clone()

        # cost volume construction
        features_mv_reshaped = rearrange(features_mv_transformed, "(b v) c h w -> b v c h w", v=v, b=b)
        features_mv_warped, intr_warped, poses_warped = (
            prepare_feat_proj_data_lists(
                features_mv_reshaped,
                intrinsics,
                extrinsics,
                num_reference_views=num_reference_views,
                idx=idx.clone()  # clone idx 以避免就地操作
            )
        )

        # 构建视差候选，避免就地操作
        min_disp = rearrange(1.0 / far, "b v -> (b v) ()")  # 移除detach，保持梯度流
        max_disp = rearrange(1.0 / near, "b v -> (b v) ()")  # 移除detach，保持梯度流
        disp_range_norm = torch.linspace(0.0, 1.0, self.num_depth_candidates, device=min_disp.device)
        # 避免就地操作，使用基本运算
        disp_range_scaled = max_disp - min_disp
        disp_candi_curr = min_disp + disp_range_norm.unsqueeze(0) * disp_range_scaled
        disp_candi_curr = disp_candi_curr.type_as(features_mv_transformed)
        disp_candi_curr = repeat(disp_candi_curr, "bv d -> bv d fh fw", fh=features_mv_transformed.shape[-2],
                                 fw=features_mv_transformed.shape[-1])  # [bxv, d, 1, 1]

        raw_correlation_in = []
        for i in range(num_reference_views):
            features_mv_warped_i = warp_with_pose_depth_candidates(
                features_mv_warped[:, i, :, :, :].clone(),  # clone 以避免就地操作
                intr_warped[:, i, :, :].clone(),
                poses_warped[:, i, :, :].clone(),
                1 / disp_candi_curr.clone(),
                warp_padding_mode="zeros"
            )

            correlation = torch.sum(
                features_mv_transformed.unsqueeze(2).clone() * features_mv_warped_i.clone(),
                dim=1
            ) / (features_mv_transformed.shape[1] ** 0.5)
            raw_correlation_in.append(correlation.clone())

        raw_correlation_in = torch.mean(torch.stack(raw_correlation_in, dim=1), dim=1).clone()

        # refine cost volume and get depths
        features_mono_tmp = F.interpolate(features_mono, (64, 64), mode="bilinear", align_corners=True).clone()
        raw_correlation_in_combined = torch.cat((
            raw_correlation_in.clone(),
            features_mv_transformed.clone(),
            features_mono_tmp.clone()
        ), dim=1).clone()

        raw_correlation = self.corr_refine_net(raw_correlation_in_combined.clone()).clone()
        raw_correlation_residual = self.regressor_residual(
            raw_correlation_in_combined.clone()
        ).clone()
        raw_correlation = (raw_correlation + raw_correlation_residual).clone()

        pdf = F.softmax(self.depth_head_lowres(raw_correlation.clone()).clone(), dim=1).clone()
        disps_metric = (disp_candi_curr * pdf).sum(dim=1, keepdim=True).clone()
        pdf_max = torch.max(pdf, dim=1, keepdim=True)[0].clone()
        pdf_max = F.interpolate(pdf_max, (ori_h, ori_w), mode="bilinear", align_corners=True).clone()
        disps_metric_fullres = F.interpolate(disps_metric, (ori_h, ori_w), mode="bilinear",
                                             align_corners=True).clone()

        # feature refinement
        features_mv_in_fullres = F.interpolate(features_mv_transformed, (ori_h, ori_w), mode="bilinear",
                                               align_corners=True).clone()
        features_mv_in_fullres = self.proj_feature_mv(features_mv_in_fullres.clone()).clone()

        features_mono_in_fullres = F.interpolate(features_mono, (ori_h, ori_w), mode="bilinear",
                                                 align_corners=True).clone()
        features_mono_in_fullres = self.proj_feature_mono(features_mono_in_fullres.clone()).clone()

        disps_rel_fullres = F.interpolate(disps_rel, (ori_h, ori_w), mode="bilinear",
                                          align_corners=True).clone()

        images_reorder = rearrange(images, "b v c h w -> (b v) c h w").clone()

        # 构建UNet输入
        unet_input = torch.cat([
            features_mv_in_fullres.clone(),
            features_mono_in_fullres.clone(),
            images_reorder.clone(),
            disps_metric_fullres.clone(),
            disps_rel_fullres.clone(),
            pdf_max.clone()
        ], dim=1).clone()

        refine_out = self.refine_unet(unet_input.clone()).clone()

        # gaussians head
        gaussians_input = torch.cat([
            refine_out.clone(),
            features_mv_in_fullres.clone(),
            features_mono_in_fullres.clone(),
            images_reorder.clone()
        ], dim=1).clone()
        raw_gaussians = self.to_gaussians(gaussians_input.clone()).clone()

        # delta fine depth and density
        disparity_input = torch.cat([
            refine_out.clone(),
            disps_metric_fullres.clone(),
            disps_rel_fullres.clone(),
            pdf_max.clone()
        ], dim=1).clone()
        delta_disps_density = self.to_disparity(disparity_input.clone()).clone()
        delta_disps, raw_densities = delta_disps_density.split(gaussians_per_pixel, dim=1)

        # outputs
        far_rearranged = rearrange(far.clone(), "b v -> (b v) () () ()")  # clone far
        near_rearranged = rearrange(near.clone(), "b v -> (b v) () () ()")  # clone near

        fine_disps = torch.clamp(
            disps_metric_fullres.clone() + delta_disps.clone(),
            1.0 / far_rearranged.clone(),
            1.0 / near_rearranged.clone(),
        ).clone()

        depths = 1.0 / fine_disps.clone()
        depths = repeat(
            depths.clone(),
            "(b v) dpt h w -> b v (h w) srf dpt",
            b=b,
            v=v,
            srf=1,
        ).clone()

        densities = repeat(
            F.sigmoid(raw_densities.clone()).clone(),
            "(b v) dpt h w -> b v (h w) srf dpt",
            b=b,
            v=v,
            srf=1,
        ).clone()

        raw_gaussians = rearrange(raw_gaussians.clone(), "(b v) c h w -> b v (h w) c", v=v, b=b).clone()

        return depths, densities, raw_gaussians