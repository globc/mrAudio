CUDA_VISIBLE_DEVICES=0 python3 evaluate.py \
            --model videollama \
            --model-path $HPC_SCRATCH/mrAudio/checkpoints/VideoLLaMA/VideoLLaMA2.1-7B-AV \
            --dataset QVH \
            --video-folder $HPC_SCRATCH/mrAudio/data/qvh/videos \
            --annotation-file $HPC_SCRATCH/mrAudio/data/qvh/lavis/val.json \
            --output-file $HPC_SCRATCH/mrAudio/data/qvh/output.json \