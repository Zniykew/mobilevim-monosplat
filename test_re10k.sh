
# 10 context views
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
    mode=test \
    dataset/view_sampler=evaluation \
    checkpointing.load=/path/to/checkpoint \
    dataset.view_sampler.index_path=assets/evaluation_index_re10k_nctx10.json \
    test.compute_scores=true \
    wandb.mode=disabled

# 2 context views
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
    mode=test \
    dataset/view_sampler=evaluation \
    checkpointing.load=/path/to/checkpoint \
    dataset.view_sampler.index_path=assets/evaluation_index_re10k_nctx2.json \
    test.compute_scores=true \
    wandb.mode=disabled