import argparse
import time
import torch
import wandb
import lavis
from torch._C.cpp.nn import Module
from utils.mr_dataset import MRDataset, collate_fn
from torch.utils.data import  DataLoader
import torch.optim as optim
from utils.utils import prepare_sample
from torch.optim.lr_scheduler import LRScheduler
import torch.distributed as dist

from lavis.common.logger import MetricLogger, SmoothedValue
import annotator.uniformer.mmcv as mmcv

import logging

import os.path as osp

from lavis.common.annotator.uniformer.mmcv.runner.checkpoint import save_checkpoint ,load_checkpoint


import torch.utils.data.sampler


def eval_epoch(model, dataloader):
    model = self.unwrap_dist_model(model)
    model = self._reload_best_model(model)

    model.eval()

    results = []
    for i, samples in enumerate(tqdm(dataloader)):
        outputs = model.generate(samples)

        for qid, query, vid, target, output in zip(samples["qids"], samples["query"], samples["vid"], samples["text_output"], outputs):
            relevant_windows = moment_str_to_list(post_process(target))
            pred_relevant_windows = moment_str_to_list(post_process(output))

            out = {
                "qid": qid,
                "query": query,
                "vid": vid,
                "relevant_windows": relevant_windows,
                "pred_relevant_windows": pred_relevant_windows,
            }

            results.append(out)

    all_metrics = eval_submission(results, results)
    return all_metrics

def train_epoch(
        epoch,
        iters_per_epoch,
        model,
        dataloader,
        lr_scheduler,
        optimizer,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1):

    model.train()  ## missing


    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

    logging.info(
        "Start training epoch {}, {} iters per inner epoch.".format(
            epoch, iters_per_epoch
        )
    )

    header = "Train: data epoch: [{}]".format(epoch)

    inner_epoch = start_iters // iters_per_epoch
    header = header + "; inner epoch [{}]".format(inner_epoch)


    for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
        # if using iter-based runner, we stop after iters_per_epoch iterations.
        if i >= iters_per_epoch:
            break

        samples = next(dataloader)

        samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
        samples.update(
            {
                "epoch": inner_epoch,
                "num_iters_per_epoch": iters_per_epoch,
                "iters": i,
            }
        )

        lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

        with torch.cuda.amp.autocast():
            loss = model.forward(samples)

        scaler.scale(loss).backward()
        

        # update gradients every accum_grad_iters iterations
        if (i + 1) % accum_grad_iters == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # log to wandb
        if dist.get_rank() == 0 and wandb.run is not None:
            wandb.log(
                {
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
            )
    metric_logger.synchronize_between_processes()
    logging.info("Averaged stats: " + str(metric_logger.global_avg()))

    return {
        k: "{:.3f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }

def run_train(args):
    if args.model == "X-InstructBLIP":
        from models.xinstructblip import XInstructBLIP
        from lavis.processors.audio_processors import BeatsAudioProcessor
        from lavis.processors.alpro_processors import AlproVideoEvalProcessor
        model = XInstructBLIP(args.model_path, args.audio_encoder)
        video_processor = AlproVideoEvalProcessor(n_frms=4, image_size=224)
        audio_processor = BeatsAudioProcessor(model_name='iter3', sampling_rate=16000, n_frames=4, is_eval=False, frame_length=512)
        

    elif args.model == "VideoLLaMA":
        from models.videollama import VideoLLaMA
        model = VideoLLaMA(args.model_path)
        video_processor = model.processor
        audio_processor = None
        

    if args.dataset in ["QVH", "Charades_STA"]:
        train_dataset = MRDataset(vis_root=args.video_folder, ann_path=args.train_annotation_file, video_processor=video_processor, audio_processor=audio_processor, model=args.model)
        val_dataset = MRDataset(vis_root=args.video_folder, ann_path=args.val_annotation_file, video_processor=video_processor, audio_processor=audio_processor, model=args.model)


    train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=get_world_size(), rank=get_rank())
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, collate_fn=collate_fn)

    val_sampler = DistributedSampler(val_dataset, shuffle=False, num_replicas=get_world_size(), rank=get_rank())
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = LRScheduler(optimizer=optimizer,)
    scaler = torch.cuda.amp.GradScaler()

    best_agg_metric = 0
    best_epoch = 0
    for cur_epoch in range(args.num_epochs):
        #training phase
        train_epoch(model=model, dataloader=train_dataloader, lr_scheduler=lr_scheduler, optimizer=optimizer, scaler=scaler)
        # validation phase, every val_freq epoch
        if cur_epoch%args.val_freq == 0:
            results = eval_epoch(model, val_dataloader)
            if is_main_process():
                agg_metrics = results["brief"]["MR-full-R1-avg"]
                if agg_metrics > best_agg_metric:
                    self._save_checkpoint(cur_epoch, is_best=True)

    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help='', required=True)
    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--audio-encoder', help='', required=False)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--train-annotation-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--val-annotation-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--val-freq', help='Validation frequency.', type=int, default=1)
    parser.add_argument('--num-epochs', help='Number epochs.', type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    run_train(args)