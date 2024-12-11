import argparse
import json
import os

from utils.mr_dataset import MRDataset, collate_fn
from torch.utils.data import DataLoader
from lavis.datasets.data_utils import prepare_sample
from tqdm import tqdm

from utils.utils import convert_percentages_to_second, post_process, moment_str_to_list


def run_inference(args):

    assert args.dataset in ["QVH", "Charades_STA"]
    n_frms = 60 if args.dataset == "QVH" else 20

    if args.model == "X-InstructBLIP":
        from models.xinstructblip import XInstructBLIP
        from lavis.processors.audio_processors import BeatsAudioProcessor
        from processors.alpro_processors import AlproVideoEvalProcessor_Stamps
        model = XInstructBLIP(args.model_path, args.audio_encoder)
        video_processor = AlproVideoEvalProcessor_Stamps(n_frms=n_frms, image_size=224)
        audio_processor = BeatsAudioProcessor(model_name='iter3', sampling_rate=16000, n_frames=n_frms, is_eval=True, frame_length=512)
        

    if args.model == "VideoLLaMA":
        from models.videollama import VideoLLaMA
        model = VideoLLaMA(args.model_path)
        video_processor = model.processor
        audio_processor = None
        

    dataset = MRDataset(vis_root=args.video_folder, ann_path=args.annotation_file, video_processor=video_processor, audio_processor=audio_processor, model=args.model)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    output_file = os.path.join(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out_file = open(output_file, "w")

    for i, samples in enumerate(tqdm(dataloader)):
        samples = prepare_sample(samples, cuda_enabled=True)
        outputs = model.generate(samples)

        for qid, query, vid, raw_out in zip(samples["qid"], samples["query"], samples["vid"], outputs):

            pred_relevant_windows = moment_str_to_list(post_process(raw_out))

            out = {
                "qid": qid,
                "query": query,
                "vid": vid,
                "pred_relevant_windows": pred_relevant_windows,
                "raw_out": raw_out
            }
        
            out_file.write(json.dumps(out) + "\n")

    out_file.close()


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
    parser.add_argument("--batch-size", type=int, required=False, default=2)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    run_inference(args)