export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
echo $CUDA_VISIBLE_DEVICES

export NUM_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
echo $NUM_GPUS


python -m torch.distributed.run --nproc_per_node=$NUM_GPUS finetune.py \
            --model X-InstructBLIP \
            --model-path $HPC_SCRATCH/mrAudio/checkpoints/X-InstructBLIP/vicuna-7b-v1.1 \
            --audio-encoder $HPC_SCRATCH/mrAudio/checkpoints/X-InstructBLIP/BEATs_iter3_plus_AS2M.pt \
            --dataset QVH \
            --video-folder $HPC_SCRATCH/mrAudio/data/qvh/videos \
            --train-annotation-file $HPC_SCRATCH/mrAudio/data/qvh/highlight_train_release.jsonl \
            --val-annotation-file $HPC_SCRATCH/mrAudio/data/qvh/highlight_val_release.jsonl \
            --output-dir $HPC_SCRATCH/mrAudio/checkpoints/qvh/X-InstructBLIP/