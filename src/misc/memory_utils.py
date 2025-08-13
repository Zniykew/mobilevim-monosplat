# src/misc/memory_utils.py
import torch

def check_memory_usage(stage=""):
    """检查CUDA内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[{stage}] CUDA Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max Allocated: {max_allocated:.2f}GB")
