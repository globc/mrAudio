import argparse
import json
from torch.utils.data import Dataset, DataLoader
from videollama2 import model_init, mm_infer
from tqdm import tqdm
import os
import ffmpeg
from tqdm import tqdm
from utils.utils import post_process, moment_str_to_list

class MRDataset(Dataset):
    def __init__(self, processor, vis_root, ann_path):
        self.processor = processor
        self.vis_root = vis_root


        with open(ann_path, "r") as f:
            self.annotation = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        video_path = os.path.join(self.vis_root, ann["vid"] + ".mp4")
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

        query_prompt = "Query: " + query + "\n"

        format = """[[x, y],[a,b]] where x,y are the first valid frames of the given query and a,b are also valid frames.
          if there is only one valid frame use [[x,y]]

          All entries must be a positive number
          Do not answer in any other format
        """

        explaination = "where relevant_windows are the actual frames that are relevant for the given query"

        example = """
query: some military patriots takes us through their safety procedures and measures. duration: 150 relevant_windows: [[72, 82], [84, 94], [96, 106], [108, 118], [120, 130], [136, 142], [144, 146]]
query: Man in baseball cap eats before doing his interview. duration: 150 relevant_windows: [[96, 114]]
query: A man in a white shirt discusses the right to have and carry firearms. duration: 150 relevant_windows: [[48, 50], [76, 120], [122, 138], [140, 146]]
query: A view of a bamboo fountain of water in a tea house and people scoop from and wash off duration: 150 relevant_windows: [[64, 92]]

"""


        task_prompt = f"""
        do not hallucinate

        Given the following example: {example}

        Explaination: {explaination}

        Please follow the examples and answer only in the following format: {format}

        The video and the query, find the relevant windows.

        Relevant windows:
        """

        text_input = query_prompt + "\n" + task_prompt

        return {
            "video": video_input,
            "text": text_input,
            # qid, query & vid necessary for QVH submission
            "qid": ann["qid"],
            "query": query,
            "vid": ann["vid"],
        }
    
def collate_fn(batch):
    video  = [x['video'] for x in batch]
    txt = [x['text'] for x in batch]
    qid  = [x['qid'] for x in batch]
    query = [x['query'] for x in batch]
    vid = [x['vid'] for x in batch]

    return video, txt, qid, query, vid

def run_inference(args):
    if "VideoLLaMA" in args.model_path:
        model, processor, tokenizer = model_init(args.model_path)

    if args.dataset in ["QVH", "Charades_STA"]:
        dataset = MRDataset(processor=processor['video'], vis_root=args.video_folder, ann_path=args.annotation_file)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    output_file = os.path.join(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out_file = open(output_file, "w")

    for i, (video_tensors, texts, qids, queries, vids) in enumerate(tqdm(dataloader)):
        video_tensor = video_tensors[0]
        text = texts[0]

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

        pred_relevant_windows = moment_str_to_list(post_process(output))

        out = {
                "qid": qids[0],
                "query": queries[0],
                "vid": vids[0],
                "pred_relevant_windows": pred_relevant_windows,
                # "pred_saliency_scores": , # TODO for QVH submission?
            }
        
        out_file.write(json.dumps(out) + "\n")

    out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
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