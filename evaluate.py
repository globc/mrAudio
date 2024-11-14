import argparse
import json
from torch.utils.data import Dataset, DataLoader
from videollama2 import model_init, mm_infer
from tqdm import tqdm
import os
import ffmpeg
from tqdm import tqdm

class MRDataset(Dataset):
    def __init__(self, processor, vis_root, ann_path):
        self.processor = processor
        self.vis_root = vis_root


        self.annotation = json.load(open(ann_path, "r"))

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        video_path = os.path.join(self.vis_root, ann["video"] + ".mp4")
        if "start" in ann:
            start, end = float(ann["start"]), float(ann["end"])

            try:
                stream = ffmpeg.input(video_path)
                stream = ffmpeg.filter(stream, 'crop', start=start, end=end)
                output_path = os.path.join(self.vis_root, f"{ann['video']}_clipped.mp4")
                ffmpeg.output(stream, output_path).run()
                video_path = output_path
            except:
                print("video read error")
                video_input = None

        try:
            video_input = self.processor(video_path, va=True)
        except:
            print("video read error")
            video_input = None

        query = ann["query"]
        relevant_windows = str(ann["relevant_windows"])

        query_prompt = "Query: " + query + "\n"
        task_prompt = "Given the video and the query, find the relevant windows.\nRelevant windows: "

        text_input = query_prompt + task_prompt

        return {
            "video": video_input,
            "text": text_input,
            "query_id": ann["qid"],
            "relevant_windows": relevant_windows,
        }
    
def collate_fn(batch):
    vid  = [x['video'] for x in batch]
    txt = [x['text'] for x in batch]
    qid  = [x['query_id'] for x in batch]
    rel_win  = [x['relevant_windows'] for x in batch]
    return vid, txt, qid, rel_win

def run_inference(args):
    if args.model == "videollama":
        model, processor, tokenizer = model_init("/work/scratch/kurse/kurs00079/cb14syta/mr-Audio2/checkpoints/VideoLLaMA/VideoLLaMA2.1-7B-AV")

    if args.dataset == "QVH":
        dataset = MRDataset(processor=processor['video'], vis_root=args.video_folder, ann_path=args.annotation_file)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    output_file = os.path.join(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out_file = open(output_file, "w")

    for i, (vid_tensor, texts, query_ids, relevant_windows) in enumerate(tqdm(dataloader)):
        video_tensor = vid_tensor[0]
        text = texts[0]
        query_id = query_ids[0]
        relevant_windows = relevant_windows[0]

        try:
            output = mm_infer(
                video_tensor,
                text,
                model=model,
                tokenizer=tokenizer,
                modal='video',
                do_sample=False,
            )
        except:
            print("generation error")
            output = "error"

        sample_set = {'id': query_id, 'text': text, 'relevant_windows': relevant_windows, 'pred': output}
        out_file.write(json.dumps(sample_set) + "\n")

    out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--annotation-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    run_inference(args)