from videollama2 import model_init, mm_infer

class VideoLLaMA():

    def __init__(self, path):
        self.model, self.processor, self.tokenizer = model_init(path)

    def generate(self, samples):
        
        

        try:
            output = mm_infer(
                samples["video"][0],
                samples["text_input"][0],
                model=self.model,
                tokenizer=self.tokenizer,
                modal='video',
                do_sample=False,
            )
        except:
            print("generation error")
            output = "error"

        return output