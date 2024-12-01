import logging
import time


def train(self):
    start_time = time.time()
    best_agg_metric = 0
    best_epoch = 0

    self.log_config()

    # resume from checkpoint if specified
    if not self.evaluate_only and self.resume_ckpt_path is not None:
        self._load_checkpoint(self.resume_ckpt_path)

    for cur_epoch in range(self.start_epoch, self.max_epoch):
        # training phase
        if not self.evaluate_only:
            logging.info("Start training")
            train_stats = self.train_epoch(cur_epoch)
            self.log_stats(split_name="train", stats=train_stats)

        # evaluation phase
        if len(self.valid_splits) > 0:
            for split_name in self.valid_splits:
                logging.info("Evaluating on {}.".format(split_name))

                val_log = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch
                )
                if val_log is not None:
                    if is_main_process():
                        assert (
                            "agg_metrics" in val_log
                        ), "No agg_metrics found in validation log."

                        agg_metrics = val_log["agg_metrics"]
                        if agg_metrics > best_agg_metric and split_name == "val":
                            best_epoch, best_agg_metric = cur_epoch, agg_metrics

                            self._save_checkpoint(cur_epoch, is_best=True)

                        val_log.update({"best_epoch": best_epoch})
                        self.log_stats(val_log, split_name)

def train_epoch(self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1):
    
    self.model.train()



    use_amp = scaler is not None

    if not hasattr(data_loader, "__next__"):
        # convert to iterator if not already
        data_loader = iter(data_loader)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

    # if iter-based runner, schedule lr based on inner epoch.
    logging.info(
        "Start training epoch {}, {} iters per inner epoch.".format(
            epoch, iters_per_epoch
        )
    )
    header = "Train: data epoch: [{}]".format(epoch)
    if start_iters is None:
        # epoch-based runner
        inner_epoch = epoch
    else:
        # In iter-based runner, we schedule the learning rate based on iterations.
        inner_epoch = start_iters // iters_per_epoch
        header = header + "; inner epoch [{}]".format(inner_epoch)

    for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
        # if using iter-based runner, we stop after iters_per_epoch iterations.
        if i >= iters_per_epoch:
            break

        samples = next(data_loader)

        samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
        samples.update(
            {
                "epoch": inner_epoch,
                "num_iters_per_epoch": iters_per_epoch,
                "iters": i,
            }
        )

        lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = self.train_step(model=model, samples=samples)

        # after_train_step()
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # update gradients every accum_grad_iters iterations
        if (i + 1) % accum_grad_iters == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # log to wandb
        if is_main_process() and wandb.run is not None:
            wandb.log(
                {
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
            )

        # print("breaking in train_inner_loop in moment_retrieval.py")
        # break

    # after train_epoch()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info("Averaged stats: " + str(metric_logger.global_avg()))

    return {
        k: "{:.3f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
}