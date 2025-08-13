import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 导入自定义模块
from src.model.encoder.mobilevim2 import mobilevim_xxs
from src.model.encoder.costvolume.depth_predictor_multiview import DepthPredictorMultiView
from src.dataset.dataset_nyu import NYUDepthV2Dataset, DatasetNYUCfg
from src.dataset.dataset_kitti import KITTIDataset, DatasetKITTICfg
from src.dataset.view_sampler import ViewSampler, ViewSamplerCfg
from src.loss.loss_depth import LossDepth as DepthLoss

# 定义默认配置
@dataclass
class DefaultDatasetCfg:
    image_shape: tuple = (384, 512)
    background_color: tuple = (0, 0, 0)
    cameras_are_circular: bool = False
    overfit_to_scene: str = None
    baseline_epsilon: float = 0.01
    max_fov: float = 120
    make_baseline_1: bool = True
    augment: bool = True
    test_len: int = 0
    test_chunk_interval: int = 1
    test_times_per_scene: int = 1
    shuffle_val: bool = True
    near: float = 0.1
    far: float = 10.0  # NYU默认值，KITTI会覆盖

# 定义默认视图采样器配置
@dataclass
class DefaultViewSamplerCfg:
    num_context_views: int = 2
    num_target_views: int = 1
    scene_scale: float = 1.0
    enable_cache: bool = True

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mobilevim_depth_train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MobileViM Depth Training')

def parse_args():
    parser = argparse.ArgumentParser(description='MobileViM Depth Estimation Pre-training')
    parser.add_argument('--dataset', type=str, default='nyu', choices=['nyu', 'kitti'],
                        help='Dataset to use for pre-training')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    return parser.parse_args()

def get_dataset(dataset_name, data_root):
    # 创建视图采样器
    view_sampler_cfg = DefaultViewSamplerCfg()
    view_sampler = ViewSampler(cfg=view_sampler_cfg)
    
    # 创建数据集配置
    default_cfg = DefaultDatasetCfg()
    
    if dataset_name == 'nyu':
        # NYU数据集配置
        dataset_cfg = DatasetNYUCfg(
            name='nyu',
            roots=[Path(data_root)],
            image_shape=default_cfg.image_shape,
            background_color=default_cfg.background_color,
            cameras_are_circular=default_cfg.cameras_are_circular,
            overfit_to_scene=default_cfg.overfit_to_scene,
            baseline_epsilon=default_cfg.baseline_epsilon,
            max_fov=default_cfg.max_fov,
            make_baseline_1=default_cfg.make_baseline_1,
            augment=default_cfg.augment,
            test_len=default_cfg.test_len,
            test_chunk_interval=default_cfg.test_chunk_interval,
            test_times_per_scene=default_cfg.test_times_per_scene,
            shuffle_val=default_cfg.shuffle_val,
            near=default_cfg.near,
            far=default_cfg.far,
            download=True
        )
        
        # 创建训练和验证数据集
        train_dataset = NYUDepthV2Dataset(
            cfg=dataset_cfg,
            stage='train',
            view_sampler=view_sampler
        )
        val_dataset = NYUDepthV2Dataset(
            cfg=dataset_cfg,
            stage='val',
            view_sampler=view_sampler
        )
    elif dataset_name == 'kitti':
        # KITTI数据集配置
        dataset_cfg = DatasetKITTICfg(
            name='kitti',
            roots=[Path(data_root)],
            image_shape=default_cfg.image_shape,
            background_color=default_cfg.background_color,
            cameras_are_circular=default_cfg.cameras_are_circular,
            overfit_to_scene=default_cfg.overfit_to_scene,
            baseline_epsilon=default_cfg.baseline_epsilon,
            max_fov=default_cfg.max_fov,
            make_baseline_1=default_cfg.make_baseline_1,
            augment=default_cfg.augment,
            test_len=default_cfg.test_len,
            test_chunk_interval=default_cfg.test_chunk_interval,
            test_times_per_scene=default_cfg.test_times_per_scene,
            shuffle_val=default_cfg.shuffle_val,
            near=default_cfg.near,
            far=80.0  # KITTI的深度范围更大
        )
        
        # 创建训练和验证数据集
        train_dataset = KITTIDataset(
            cfg=dataset_cfg,
            stage='train',
            view_sampler=view_sampler
        )
        val_dataset = KITTIDataset(
            cfg=dataset_cfg,
            stage='val',
            view_sampler=view_sampler
        )
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return train_dataset, val_dataset

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(dataloader):
        # 从batch中提取目标视图的数据
        images = batch['target']['image'].to(device)
        depths = batch['target']['depth'].to(device)
        intrinsics = batch['target']['intrinsics'].to(device)
        extrinsics = batch['target']['extrinsics'].to(device)
        near = batch['target']['near'].to(device)
        far = batch['target']['far'].to(device)

        # 提取上下文视图的数据
        context_images = batch['context']['image'].to(device)
        context_intrinsics = batch['context']['intrinsics'].to(device)
        context_extrinsics = batch['context']['extrinsics'].to(device)

        # 合并上下文和目标视图的数据
        all_images = torch.cat([context_images, images.unsqueeze(1)], dim=1)
        all_intrinsics = torch.cat([context_intrinsics, intrinsics.unsqueeze(1)], dim=1)
        all_extrinsics = torch.cat([context_extrinsics, extrinsics.unsqueeze(1)], dim=1)

        # Forward pass
        optimizer.zero_grad()
        output = model(all_images, all_intrinsics, all_extrinsics, near, far)
        loss = criterion(output['disps_metric_fullres'], depths)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 100 == 0:
            logger.info(f'Epoch {epoch}, Batch {i+1}, Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(dataloader)
    logger.info(f'Epoch {epoch}, Average Training Loss: {avg_loss:.4f}')
    return avg_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            # 从batch中提取目标视图的数据
            images = batch['target']['image'].to(device)
            depths = batch['target']['depth'].to(device)
            intrinsics = batch['target']['intrinsics'].to(device)
            extrinsics = batch['target']['extrinsics'].to(device)
            near = batch['target']['near'].to(device)
            far = batch['target']['far'].to(device)

            # 提取上下文视图的数据
            context_images = batch['context']['image'].to(device)
            context_intrinsics = batch['context']['intrinsics'].to(device)
            context_extrinsics = batch['context']['extrinsics'].to(device)

            # 合并上下文和目标视图的数据
            all_images = torch.cat([context_images, images.unsqueeze(1)], dim=1)
            all_intrinsics = torch.cat([context_intrinsics, intrinsics.unsqueeze(1)], dim=1)
            all_extrinsics = torch.cat([context_extrinsics, extrinsics.unsqueeze(1)], dim=1)

            output = model(all_images, all_intrinsics, all_extrinsics, near, far)
            loss = criterion(output['disps_metric_fullres'], depths)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    logger.info(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss

def main():
    args = parse_args()
    device = torch.device(args.device)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置TensorBoard
    log_dir = os.path.join('runs', f'mobilevim_depth_{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    writer = SummaryWriter(log_dir)

    # 加载数据集
    train_dataset, val_dataset = get_dataset(args.dataset, args.data_root)
    # 使用IterableDataset时，shuffle参数会被忽略，num_workers应设置为0
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    # 创建模型
    mobilevim_model = mobilevim_xxs(pretrained=False)
    model = DepthPredictorMultiView(
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
        ffn_dim_expansion=4
    )

    # 替换模型中的mobilevim
    model.mobilevim = mobilevim_model

    # 只训练mobilevim和相关投影层
    for param in model.parameters():
        param.requires_grad = False
    for param in model.mobilevim.parameters():
        param.requires_grad = True
    for param in model.feature_projections.parameters():
        param.requires_grad = True
    for param in model.depth_head.parameters():
        param.requires_grad = False  # 保持depth_head冻结
    for param in model.cost_head.parameters():
        param.requires_grad = True

    model = model.to(device)

    # 损失函数和优化器
    criterion = DepthLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 恢复训练
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f'Resumed from checkpoint: {args.resume}, starting at epoch {start_epoch}')

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_dataloader, criterion, device)

        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.save_dir, f'mobilevim_depth_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, checkpoint_path)
            logger.info(f'Saved best model to {checkpoint_path}')

        # 保存每个epoch的模型
        checkpoint_path = os.path.join(args.save_dir, f'mobilevim_depth_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)

        # 更新学习率
        scheduler.step()

    logger.info('Training completed!')
    writer.close()

if __name__ == '__main__':
    main()