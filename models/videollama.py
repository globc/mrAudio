from videollama2 import model_init, mm_infer

class VideoLLaMA():

    def __init__(self, path):
        self.model, self.processor, self.tokenizer = model_init(path)

    def generate(self, video_paths, texts):
        
        try:
            video_tensor = self.processor(video_paths[0], va=True)
        except:
            print("video read error")
            video_tensor = None

        try:
            output = mm_infer(
                video_tensor,
                texts[0],
                model=self.model,
                tokenizer=self.tokenizer,
                modal='video',
                do_sample=False,
            )
        except:
            print("generation error")
            output = "error"

        return output