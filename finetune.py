import argparse
import time
import torch
import wandb
from torch._C.cpp.nn import Module
from mrAudio.evaluate import MRDataset, collate_fn
from torch.utils.data import  DataLoader
import torch.optim as optim
from mrAudio.utils.utils import prepare_sample
from torch.optim.lr_scheduler import LRScheduler
import torch.distributed as dist

def train(model,start_epoch:int,max_epoch:int,valid_splits:[],val_freq:int,data_loader:DataLoader,evaluate_only:bool=False):
    start_time = time.time()
    best_agg_metric = 0
    best_epoch = 0

    """
    learning_rate = 0.001  # Adjust as needed
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    """
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = LRScheduler(optimizer=optimizer,)

    # resume from checkpoint if specified
    ##if not self.evaluate_only and self.resume_ckpt_path is not None:
    ##    self._load_checkpoint(self.resume_ckpt_path)

    for cur_epoch in range(start_epoch, max_epoch):
        if len(valid_splits) > 0 and (evaluate_only or cur_epoch%val_freq == 0):
            for split_name in valid_splits:
                train_epoch(epoch=cur_epoch,model=model,iters_per_epoch=int(len(valid_splits)/val_freq),data_loader=data_loader,lr_scheduler=lr_scheduler,optimizer=optimizer)


def train_epoch(
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        lr_scheduler,
        optimizer,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1):
    
    model.train()  ## missing

    losses = []
    for i in range(iters_per_epoch):
        samples = next(data_loader)
        if start_iters is None:
            inner_epoch = epoch
        else:

            inner_epoch = start_iters // iters_per_epoch

        samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
        samples.update(
            {
                "epoch": inner_epoch,
                "num_iters_per_epoch": iters_per_epoch,
                "iters": i,
            }
        )



        lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

        with torch.cuda.amp.autocast(enabled=True):
            loss = model.forward(**samples)
            losses.append(loss)

        scaler.scale(loss).backward()


        if (i + 1) % accum_grad_iters == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


        # log to wandb
        if dist.get_rank() == 0 and wandb.run is not None:
            wandb.log(
                {
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
            )
    return losses

def run_train(args):

    model:Module
    if args.model == "X-InstructBLIP":
        from models.xinstructblip import XInstructBLIP
        model = XInstructBLIP(args.model_path, args.audio_encoder)

    if args.model == "VideoLLaMA":
        from models.videollama import VideoLLaMA
        model = VideoLLaMA(args.model_path)

    dataset:MRDataset
    if args.dataset in ["QVH", "Charades_STA"]:
        dataset = MRDataset(vis_root=args.video_folder, ann_path=args.annotation_file)

    dataloader = DataLoader(dataset=dataset,
                            shuffle=False,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn)


    train(model,0,4,0,1,dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help='', required=True)
    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--audio-encoder', help='', required=False)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--annotation-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    run_train(args)