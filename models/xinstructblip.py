from lavis.processors.audio_processors import BeatsAudioProcessor
from lavis.processors.alpro_processors import AlproVideoEvalProcessor
from omegaconf import OmegaConf
from lavis.common.registry import registry



class XInstructBLIP():
    def __init__(self, model_path, audio_path):
        self.audio_processor = BeatsAudioProcessor(model_name='iter3', sampling_rate=16000, n_frames=2, is_eval=False, frame_length=512)
        self.video_processor = AlproVideoEvalProcessor(n_frms=4, image_size=224)

        # from lavis.models import load_model
        # self.model = load_model("blip2_vicuna_xinstruct", "vicuna7b")

        self.config = OmegaConf.load("./models/vicuna7b_v2.yaml")
        self.config.get("model", None).llm_model = model_path
        self.config.get("model", None).audio_encoder_kwargs = {"checkpoint_path": audio_path}
        model_cls = registry.get_model_class(self.config.get("model", None).arch)
        self.model =  model_cls.from_config(self.config.get("model", None))
        self.model.to("cuda")

    def generate(self, video_paths, texts):
        prompt = texts[0]
        video_path = video_paths[0]


        audio = self.audio_processor(video_path).unsqueeze(0).to("cuda")
        video = self.video_processor(video_path).unsqueeze(0).to("cuda")
        
        samples = {"prompt": prompt, "audio": audio, "video": video}

        output = self.model.generate(
            samples
        )

        return output[0]