
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dtu \
    mode=test \
    checkpointing.load=/path/to/checkpoint \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.index_path=assets/evaluation_index_dtu_nctx2.json \
    test.compute_scores=true \
    wandb.mode=disabled