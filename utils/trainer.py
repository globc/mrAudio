import logging
import torch
import os
from tqdm import tqdm
import torch.distributed as dist
from lavis.common.dist_utils import download_cached_file
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.optims import LinearWarmupCosineLRScheduler
from lavis.common.utils import is_url
from lavis.datasets.data_utils import prepare_sample
from torch.distributed import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader

from eval.mr_eval import eval_submission
from utils.mr_dataset import MRDataset, collate_fn
from utils.utils import moment_str_to_list, post_process


class Trainer:
    def __init__(
            self,
            args,
            ):
        self.val_freq = args.val_freq
        self.save_freq = args.save_freq
        self.max_epoch = args.max_epoch
        self.output_dir = args.output_dir
        self.resume_ckpt_path = None # Change if resume
        self.device = torch.device("cuda")
        self.accum_grad_iters = 2
        self.start_epoch = 0

        assert args.dataset in ["QVH", "Charades_STA"]
        n_frms = 60 if args.dataset == "QVH" else 20

        # get model
        if args.model == "X-InstructBLIP":
            from models.xinstructblip import XInstructBLIP
            self.model = XInstructBLIP(args.model_path, args.audio_encoder)
            from processors.alpro_processors import AlproVideoTrainProcessor_Stamps
            from lavis import BeatsAudioProcessor
            from processors.alpro_processors import AlproVideoEvalProcessor_Stamps
            train_video_processor = AlproVideoTrainProcessor_Stamps(n_frms=n_frms, image_size=224)
            val_video_processor = AlproVideoEvalProcessor_Stamps(n_frms=n_frms, image_size=224)
            audio_processor = BeatsAudioProcessor(model_name='iter3', sampling_rate=16000, n_frames=n_frms, is_eval=False, frame_length=512)
        

        elif args.model == "VideoLLaMA":
            from models.videollama import VideoLLaMA
            self.model = VideoLLaMA(args.model_path)
            train_video_processor = self.model.processor
            val_video_processor = self.model.processor
            audio_processor = None
            
        self.model = self.model.to(args.gpu)

        self.optimizer = optim.optim.Adam(self.model.parameters(), lr=3e-4)
        self.lr_scheduler = LinearWarmupCosineLRScheduler(self.optimizer, self.max_epoch, min_lr=0, init_lr=3e-4, warmup_steps=2255, warmup_start_lr=1e-8)
        self.scaler = torch.cuda.amp.GradScaler()

        self.model = DistributedDataParallel(self.model, device_ids=[args.gpu])

        train_dataset = MRDataset(vis_root=args.video_folder, ann_path=args.train_annotation_file, video_processor=train_video_processor, audio_processor=audio_processor, model=args.model)
        val_dataset = MRDataset(vis_root=args.video_folder, ann_path=args.val_annotation_file, video_processor=val_video_processor, audio_processor=audio_processor, model=args.model)

        train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        val_sampler = DistributedSampler(val_dataset, shuffle=False, num_replicas=dist.get_world_size(), rank=dist.get_rank())

        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, collate_fn=collate_fn)
        self.val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, collate_fn=collate_fn)

    def train(self):
        best_agg_metric = 0
        best_epoch = 0

        # resume from checkpoint if specified
        if self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            #training phase
            self.train_epoch(cur_epoch)
            # validation phase, every val_freq epoch
            if cur_epoch%self.val_freq == 0:
                results = self.eval_epoch()
                if dist.get_rank() == 0: # only need to save once
                    agg_metrics = results["brief"]["MR-full-R1-avg"]
                    logging.info("MR performance at epoch {}: {}".format(cur_epoch, agg_metrics))
                    if agg_metrics > best_agg_metric:
                        best_epoch, best_agg_metric = cur_epoch, agg_metrics
                        self._save_checkpoint(cur_epoch, is_best=True)

            # save checkpoint according to save freq
            if self.save_freq>0 and cur_epoch%self.save_freq == 0:
                self._save_checkpoint(cur_epoch, is_best=False)


        dist.barrier()

    def train_epoch(self, cur_epoch):

        self.model.train()  ## missing


        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                cur_epoch, len(self.train_dataloader)
            )
        )


        for i, samples in enumerate(tqdm(self.train_dataloader)):
            samples = prepare_sample(samples, cuda_enabled=True)

            self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=i)

            with torch.cuda.amp.autocast():
                output = self.model(samples)
                loss = output["loss"] / self.accum_grad_iters

            self.scaler.scale(loss).backward()
            

            # update gradients every accum_grad_iters iterations
            if (i + 1) % self.accum_grad_iters == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()


            loss_dict = {k: v for k, v in output.items() if "loss" in k}
            metric_logger.update(**loss_dict)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])


        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))

        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @torch.no_grad()
    def eval_epoch(self):
        model = self.model.module # unwrap from DDP

        model.eval()

        results = []
        for i, samples in enumerate(tqdm(self.val_dataloader)):
            samples = prepare_sample(samples, cuda_enabled=True)
            outputs = model.generate(samples)

            for qid, query, vid, target, output in zip(samples["qid"], samples["query"], samples["vid"], samples["text_output"], outputs):
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
    
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model = self.model.module # unwrap from DDP to access parameters
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model.named_parameters()
        }
        state_dict = model.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        os.makedirs(self.output_dir, exist_ok=True)
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model
    
    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.model.module.load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))