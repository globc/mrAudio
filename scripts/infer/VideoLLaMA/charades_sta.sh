ANNOTATION_FILE=$HPC_SCRATCH/mrAudio/data/Charades/Charades_STA/proc/new_val.jsonl
OUTPUT_FILE=$HPC_SCRATCH/mrAudio/data/Charades/Charades_STA/output.jsonl
RESULT_FILE=$HPC_SCRATCH/mrAudio/data/Charades/Charades_STA/results.json

CUDA_VISIBLE_DEVICES=0 python3 evaluate.py \
            --model VideoLLaMA \
            --model-path $HPC_SCRATCH/mrAudio/checkpoints/VideoLLaMA/VideoLLaMA2.1-7B-AV \
            --dataset Charades_STA \
            --video-folder $HPC_SCRATCH/mrAudio/data/Charades/Charades_v1 \
            --annotation-file ${ANNOTATION_FILE} \
            --output-file ${OUTPUT_FILE} \

cd eval
python3 mr_eval.py \
    --submission_path ${OUTPUT_FILE} \
    --gt_path ${ANNOTATION_FILE} \
    --save_path ${RESULT_FILE} \