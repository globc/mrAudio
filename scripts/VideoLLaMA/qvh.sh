CUDA_VISIBLE_DEVICES=0 python3 evaluate.py \
            --model videollama \
            --dataset QVH \
            --video-folder /work/scratch/kurse/kurs00079/cb14syta/mr-Audio2/data/QVH/videos \
            --annotation-file /work/scratch/kurse/kurs00079/cb14syta/mr-Audio2/data/QVH/lavis/val.json \
            --output-file /work/scratch/kurse/kurs00079/cb14syta/mr-Audio2/outputs/QVH/videollama/output.json \