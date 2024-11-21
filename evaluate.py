import argparse
import json
from torch.utils.data import Dataset, DataLoader
from videollama2 import model_init, mm_infer
from tqdm import tqdm
import os
import ffmpeg
from tqdm import tqdm
from utils.utils import convert_percentages_to_second, post_process, moment_str_to_list

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
                ffmpeg.output(stream, output_path)
                ffmpeg.run(stream,overwrite_output=True)
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

        example = """"
                    query: <Query> some military patriots takes us through their safety procedures and measures. <Query> 
                    duration: <Duration> 150 </Duration>
                    relevant_windows: [[0.80, 0.83], [0.84, 0.94]]', 

                    query: <Query> Man in baseball cap eats before doing his interview. <Query> 
                    duration:  <Duration> 150  </Duration> 
                    relevant_windows: [[0.96, 1]]'

                    query: <Query> A view of a bamboo fountain of water in a tea house and people scoop from and wash off <Query> 
                    duration:  <Duration> 150  </Duration> 
                    relevant_windows: [[0.21, 0.99]]'

                    query: <Query> The weather map shows snowfall <Query> 
                    duration:  <Duration> 150  </Duration> 
                    relevant_windows: [[0.12, 0.17],[0.27, 0.30],[0.32, 0.42],[0.43, 0.50],[0.68, 0.70],[0.80, 0.82]]'
                """


        format_text = """[[x, y],[a,b],[c,d]]
            if there is only one valid frame use [[x,y]]
            they represent the persentages of the video duration
            Ensure that the windows are in ascending order and do not overlap.
        """

        prompt = f"""
        Do not hallucinate \n
        follow the flowing text as accurate as possible \n

        Example: <Example> {example} </Example> \n
        Format: <Format> {format_text} </Format> \n
        Query: <Query> {query} </Query> 
        Duration: <Duration> {ann["duration"]} </Duration> \n

        For die video give me the relevant windows matching the Query for the given duration \n
        relevant_windows:  \n
        """

        text_input = prompt 

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

        raw_out = output
        output = convert_percentages_to_second(output,150)

        pred_relevant_windows = moment_str_to_list(post_process(output))

        out = {
                "raw_out": raw_out,
                "qid": qids[0],
                "query": queries[0],
                "vid": vids[0],
                "pred_relevant_windows": pred_relevant_windows,
                "raw_out": raw_out,
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