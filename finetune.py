import argparse
from utils.trainer import Trainer
import datetime
import os
import torch
import torch.distributed as dist


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()


    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    dist.barrier()

    import builtins as __builtin__

    builtin_print = __builtin__.print

    is_master = args.rank == 0
    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def run_train(args):
    init_distributed_mode(args)
    trainer = Trainer(args)
    trainer.train()
    dist.destroy_process_group()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help='', required=True)
    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--audio-encoder', help='', required=False)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--train-annotation-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--val-annotation-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output-dir', help='Directory to save the model checkpoints.', required=True)
    parser.add_argument('--val-freq', help='Validation frequency.', type=int, default=1)
    parser.add_argument('--save-freq', help='Checkpoint save frequency.', type=int, default=1)
    parser.add_argument('--max-epoch', help='Number epochs.', type=int, default=50)
    parser.add_argument('--lr', help='Learning rate.', type=float, default=1e-3)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, required=False, default=2)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    run_train(args)