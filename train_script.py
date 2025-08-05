#!/usr/bin/env python3
import subprocess
import sys
import os


def main():
    # 设置环境变量以优化内存使用
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    env['HYDRA_FULL_ERROR'] = '1'  # 启用完整错误追踪
    env['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用CUDA调试模式

    # 设置wandb API密钥
    env['WANDB_API_KEY'] = '73f6d2ab7d90d2ab9f213d25dc1af0821a5074ed'

    # 构建训练命令 - 使用conda activate
    cmd = [
        'conda', 'run', '-n', 'MonoSplat', 'python', '-m', 'src.main',
        '+experiment=re10k',
        'data_loader.train.batch_size=4',
        '+model.decoder.max_gaussian_points=16384',
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
