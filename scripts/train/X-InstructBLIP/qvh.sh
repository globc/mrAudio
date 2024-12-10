export CUDA_VISIBLE_DEVICES=nvidia-smi --list-gpus | grep GPU | awk -F: '{print $1}' | awk '{print $2}' | tr '\n' ',' | sed 's/,$//'
echo $CUDA_VISIBLE_DEVICES

python -m torch.distributed.run --nproc_per_node=4 finetune.py \

export CUDA_VISIBLE_DEVICES=nvidia-smi --list-gpus | grep GPU | awk -F: '{print $1}' | awk '{print $2}' | tr '\n' ',' | sed 's/,$//'
echo $CUDA_VISIBLE_DEVICES

python -m torch.distributed.run --nproc_per_node=4 finetune.py \
            --model X-InstructBLIP \
            --model-path $HPC_SCRATCH/mrAudio/checkpoints/X-InstructBLIP/vicuna-7b-v1.1 \
            --audio-encoder $HPC_SCRATCH/mrAudio/checkpoints/X-InstructBLIP/BEATs_iter3_plus_AS2M.pt \
            --dataset QVH \
            --video-folder $HPC_SCRATCH/mrAudio/data/qvh/videos \
            --train-annotation-file $HPC_SCRATCH/mrAudio/data/qvh/highlight_train_release.jsonl \
            --val-annotation-file $HPC_SCRATCH/mrAudio/data/qvh/highlight_val_release.jsonl \
            --output-dir $HPC_SCRATCH/mrAudio/checkpoints/qvh/X-InstructBLIP/