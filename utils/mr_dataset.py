import ffmpeg
import json
from torch.utils.data import Dataset
import os
import torch

class MRDataset(Dataset):
    def __init__(self, vis_root, ann_path, video_processor, audio_processor, model):
        self.vis_root = vis_root
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.model = model

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
                video_path = None

        if self.model == "X-InstructBLIP":
            if self.audio_processor is not None:
                audio = self.audio_processor(video_path)

            video, indices, fps = self.video_processor(video_path)

            timestamps = [round(idx / fps) for idx in indices]


        if self.model == "VideoLLaMA":
            # TODO timestamps
            try:
                video = self.video_processor(video_path, va=True)
            except:
                print("video read error")
                video = None

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

        query_prompt = "Query: " + query + "\n"
        task_prompt = "Given the video and the query, find the relevant windows.\nRelevant windows: "
        text_input = query_prompt + task_prompt

        return {
            "text_input": text_input,
            "text_output": str(ann["relevant_windows"]),
            "video": video,
            "audio": audio,
            "timestamps": timestamps,
            "duration": ann["duration"],
            # qid, query & vid necessary for QVH submission
            "qid": ann["qid"],
            "query": query,
            "vid": ann["vid"],
        }
    
def collate_fn(batch):
    collated = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        collated[key] = torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else values
    return collated
    