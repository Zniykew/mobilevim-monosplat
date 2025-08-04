
CUDA_VISIBLE_DEVICES=0 python -m src.main \
    +experiment=re10k \
    data_loader.train.batch_size=14 \
    checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth 