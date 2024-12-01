from lavis.processors.audio_processors import BeatsAudioProcessor
from lavis.processors.alpro_processors import AlproVideoEvalProcessor
from omegaconf import OmegaConf
from lavis.common.registry import registry
from transformers import LlamaForCausalLM, LlamaTokenizer



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
    

    def forward(self, video_paths, texts):
        # Freeze QFormers # TODO adapt to X-InstructBLIP (QFormer per modality),...
        for name, param in self.Qformer.named_parameters():
            param.requires_grad = False
            self.query_tokens.requires_grad = False
            self.t5_proj.requires_grad = False

        # Freeze Encoders
        for name, param in modality_encoder.named_parameters():
            param.requires_grad = False
            modality_encoder = modality_encoder.eval()
            modality_encoder.train = disabled_train


        model_modules = str(self.t5_model.modules)
        pattern = r"\((\w+)\): Linear"
        linear_layer_names = re.findall(pattern, model_modules)

        names = []
        # Print the names of the Linear layers
        for name in linear_layer_names:
            names.append(name)
        target_modules = list(set(names))


        lora_config = LoraConfig(
            r=8,
            target_modules=target_modules,
            lora_alpha=8,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm_model = get_peft_model(self.llm_model, lora_config)
        self.llm_model.gradient_checkpointing_enable()
        self.llm_model.enable_input_require_grads()
        self.llm_model.lm_head = CastOutputToFloat(self.llm_model.lm_head)
        self.llm_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        self.llm_hidden_size = self.llm_model.config.hidden_size
        self.llm_model = get_peft_model(self.llm_model, self.peft_config)

        inputs_embeds = self.llm_model.get_input_embeddings()


        outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        
        loss = dummy_loss+outputs.loss



        return {"loss": loss}