#!/usr/bin/env python3
import subprocess
import sys
import os


def main():
    # 设置环境变量以优化内存使用
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # 构建训练命令 - 使用conda activate
    cmd = [
        'conda', 'run', '-n', 'MonoSplat',  # 激活conda环境
        sys.executable, '-m', 'src.main',
        '+experiment=re10k',
        'data_loader.train.batch_size=14',
        'wandb.mode=disabled'
    ]

    print("执行训练命令:")
    print(' '.join(cmd))
    print("=" * 50)

    # 执行训练
    try:
        result = subprocess.run(cmd, env=env)
        if result.returncode == 0:
            print("\n训练成功完成!")
        else:
            print(f"\n训练失败，返回码: {result.returncode}")
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"执行训练时发生错误: {e}")


if __name__ == '__main__':
    main()
