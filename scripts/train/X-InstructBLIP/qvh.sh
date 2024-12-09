CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 finetune.py \
            --model X-InstructBLIP \
            --model-path $HPC_SCRATCH/mrAudio/checkpoints/X-InstructBLIP/vicuna-7b-v1.1 \
            --audio-encoder $HPC_SCRATCH/mrAudio/checkpoints/X-InstructBLIP/BEATs_iter3_plus_AS2M.pt \
            --dataset QVH \
            --video-folder $HPC_SCRATCH/mrAudio/data/qvh/videos \
            --train-annotation-file $HPC_SCRATCH/mrAudio/data/qvh/highlight_train_release.jsonl \
            --val-annotation-file $HPC_SCRATCH/mrAudio/data/qvh/highlight_val_release.jsonl \
            --output-file ${OUTPUT_FILE} \