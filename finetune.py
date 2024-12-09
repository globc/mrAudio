import argparse

from utils.trainer import Trainer
from torch.distributed import init_process_group

def run_train(args):
    init_process_group(backend="nccl")
    trainer = Trainer(args)
    trainer.train()
    




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
    parser.add_argument('--num-epochs', help='Number epochs.', type=int, default=50)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, required=False, default=2)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    run_train(args)


# device = 0 equal to "cuda:0"
# Wrap model with DDP(model), get underlying model with model.module (for saving checkpoint, only safe from rank 0 process)
# torchrun automatically rank, world_size. access rank with os.environ["LOCAL_RANK"]