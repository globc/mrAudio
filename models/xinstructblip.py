import logging

from lavis.processors.audio_processors import BeatsAudioProcessor
from lavis.processors.alpro_processors import AlproVideoEvalProcessor
from omegaconf import OmegaConf
from lavis.common.registry import registry
from transformers import LlamaTokenizer, BitsAndBytesConfig
from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
import torch
import random
import contextlib

from peft import prepare_model_for_kbit_training

def concat_text_input_output(input_ids, input_atts, output_ids, output_atts):
    input_part_targets_len = []
    llm_tokens = {"input_ids": [], "attention_mask": []}
    for i in range(input_ids.size(0)):
        this_input_ones = input_atts[i].sum()
        input_part_targets_len.append(this_input_ones)
        llm_tokens['input_ids'].append(
            torch.cat([
                input_ids[i][:this_input_ones],
                output_ids[i][1:],
                input_ids[i][this_input_ones:]
            ])
        )
        llm_tokens['attention_mask'].append(
            torch.cat([
                input_atts[i][:this_input_ones],
                output_atts[i][1:],
                input_atts[i][this_input_ones:]
            ])
        )
    llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
    llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
    return llm_tokens, input_part_targets_len


def maybe_autocast(self, dtype=torch.float16):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    enable_autocast = self.device != torch.device("cpu")

    if enable_autocast:
        return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return contextlib.nullcontext()


class XInstructBLIP(Blip2Base):
    
    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def __init__(self, model_path, audio_path):
        super().__init__()
        self.enumerate_inputs = False
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

        self.modalities = ["audio", "video"] # TODO set to ["video"] for baselines


        # Init video encoder
        self.pretrained_video_qformer = "https://storage.googleapis.com/sfr-xinstructblip-data-research/model/xinstructblip_checkpoints/vicuna7b/video_qformer.pth"
        video_encoder_kwargs = {
            "image_size": 224,
            "drop_path_rate": 0,
            "use_grad_checkpoint": False,
            "load_ln_path": self.pretrained_video_qformer,
            "load_ln_type": "video"
        }
        self.video_encoder, self.video_ln = self.init_video_encoder("eva_clip_g", precision="fp16", **video_encoder_kwargs)
        
        # Freeze video encoder
        for name, param in self.video_encoder.named_parameters():
            param.requires_grad = False
        self.video_encoder = self.video_encoder.eval()
        self.video_encoder.train = disabled_train


        if "audio" in self.modalities: # For baselines w/o audio
            # Init audio encoder
            self.pretrained_audio_qformer = "https://storage.googleapis.com/sfr-xinstructblip-data-research/model/xinstructblip_checkpoints/vicuna7b/video_qformer.pth"
            audio_encoder_kwargs = {
                "checkpoint_path": audio_path,
                "load_ln_path": self.pretrained_audio_qformer,
                "load_ln_type": "audio"
            }
            self.audio_encoder, self.audio_ln = self.init_audio_encoder("beats", precision="fp16", **audio_encoder_kwargs)

            # Freeze audio encoder
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder = self.audio_encoder.eval()
            self.audio_encoder.train = disabled_train



        ##### Init QFormers ####
        self.tokenizer = self.init_tokenizer(truncation_side="left") # 30523 tokens. 
        self.num_query_token = 32
        
        for modality in self.modalities:
            modality_num_features = getattr(self, f"{modality}_encoder").num_features
            
            logging.info(f"Initializing {modality} QFormer and query tokens of length {self.num_query_token}")
            modality_qformer, modality_query_tokens = self.init_Qformer(
                self.num_query_token, 
                modality_num_features,
                pretrained_qformer=getattr(self, f"pretrained_{modality}_qformer"),
                load_attention=True,
                load_qformer_type=modality
            ) 

            modality_qformer.resize_token_embeddings(len(self.tokenizer))
            modality_qformer.cls = None
            setattr(self, f"{modality}_Qformer", modality_qformer)
            setattr(self, f"{modality}_query_tokens", modality_query_tokens)

        
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False, truncation_side="left")
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})


        if self.finetune:
            from model_utils import get_peft_config
            # reduce memory usage by loading model in 4 bit quantization, allowed as model is frozen using LoRA
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.llm_model = LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                quantization_config=quantization_config
            )
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
            
            # reduce memory usage
            self.llm_model.gradient_checkpointing_enable()
            # adjust model for finetuning using low quantization
            self.llm_model = prepare_model_for_kbit_training(self.llm_model)

            self.llm_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
            self.llm_hidden_size = self.llm_model.config.hidden_size

            self.llm_model = get_peft_config(self.llm_model)

            # LLM frozen by peft by default
        
        else:
            self.llm_model = LlamaForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16
            )
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

            # Freeze LLM
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

            self.load_projection_video = True
            self.load_projection_audio = True
            self.load_projection_type_video = "video"
            self.load_projection_type_audio = "audio"
            for modality in self.modalities:
                load_projection_path = getattr(self, f"pretrained_{modality}_qformer")
                setattr(self, f"load_projection_{modality}", load_projection_path)

                qformer = getattr(self, f"{modality}_Qformer")
                proj = self.init_vicuna_projection(
                    qformer.config.hidden_size, 
                    self.llm_hidden_size,
                    load_projection_path=load_projection_path, 
                    load_projection_type=getattr(self, f"load_projection_type_{modality}")
                )
                setattr(self, f"{modality}_llm_proj", proj)
        

        self.MODALITY_TO_CUE = {
            "video": " video: ",
            "audio": " audio: ",
        }
        
        self.tokenized_cue = {}
        self.emb_cue = {}
        self.att_cue = {}
        for modality in self.modalities:
            self.tokenized_cue[modality] = self.llm_tokenizer(self.MODALITY_TO_CUE[modality], return_tensors="pt")
            self.emb_cue[modality] = self.llm_model.get_input_embeddings()(self.tokenized_cue[modality].input_ids.to(self.device))
            self.att_cue[modality] = self.tokenized_cue[modality].attention_mask.to(self.device)

        


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
    
    def forward(self,samples):
        
        if samples is None or samples == {} or not any([modality in samples for modality in self.modalities]):
            return {"loss": torch.tensor(0.0)}

        random.shuffle(self.modalities)

        curr_modalities = [modality for modality in self.modalities if modality in samples]

        # Freeze QFormers (this way higher compatibility)
        dummy_loss = 0.
        for modality in self.modalities:
            for name, param in getattr(self,f"{modality}_ln").named_parameters():
                # param.requires_grad = False
                dummy_loss += param.sum()*0.
            dummy_loss += getattr(self, f"{modality}_query_tokens").sum()*0.
            for name, param in getattr(self, f'{modality}_Qformer').named_parameters():
                    # param.requires_grad = False
                    dummy_loss += param.sum()*0.
            for name, param in getattr(self, f'{modality}_llm_proj').named_parameters():
                    # param.requires_grad = False
                    dummy_loss += param.sum()*0.
        
        
        embeds = {}
        query_tokens = {}
        data_atts = {}
        for modality in curr_modalities:
            data = samples[modality]
            ln = getattr(self, f"{modality}_ln")
            encoder = getattr(self, f"{modality}_encoder")
            if modality == "video":  
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(2)):
                    this_frame = data[:,:,j,:,:]
                    with maybe_autocast(self):
                        embeds[modality].append(ln(encoder(this_frame)))
                        data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                # B, Token Size, LM EMB
                query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(data.size(0), -1, -1)
                    
            elif modality == 'audio':
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(1)):
                    this_frame = data[:,j,:,:]
                    with maybe_autocast(self):
                        embeds[modality].append(ln(encoder(this_frame)))
                    data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                # B, Token Size, LM EMB
                query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(data.size(0), -1, -1)
        
        query_outputs = {}
        text_Qformer = self.tokenizer(
                samples["text_input"] if not self.special_qformer_input_prompt else self.special_qformer_input_prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
        
        
        Qformer_atts = {}
        query_atts = {}
        
        for modality in curr_modalities:
            # B, Token Size
            query_atts[modality] = torch.ones(query_tokens[modality].size()[:-1], dtype=torch.long).to(self.device)
            # B, Token Size + Inp Size
            Qformer_atts[modality] = torch.cat([query_atts[modality],text_Qformer.attention_mask],dim=1)
            num = len(embeds[modality])
            bs = embeds[modality][0].shape[0]
            indices = [j_+r for r,j in enumerate([[i*bs for i in range(num)]]*bs) for j_ in j]
            reordered_embeds = torch.cat(embeds[modality])[indices]
            reordered_atts = torch.cat(data_atts[modality])[indices]
            query_output = getattr(self, f"{modality}_Qformer").bert(
                text_Qformer.input_ids.repeat(num, 1),
                attention_mask=Qformer_atts[modality].repeat(num, 1),
                query_embeds=query_tokens[modality].repeat(num, 1, 1),
                encoder_hidden_states=reordered_embeds,
                encoder_attention_mask=reordered_atts,
                return_dict=True,
            )
            query_outputs[modality] = query_output
                    
        
        inputs_llm = {}
        atts_llm = {}
        for modality in curr_modalities:
                inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:]) 
                # bs, num, num query tokens, llm emb size -> bs, num*num query tokens, llm emb size
                inputs_llm[modality] = inputs_llm[modality].reshape(bs, num, self.num_query_token, -1).view(bs, num*self.num_query_token, -1)
                atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)   
        
        

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'

        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=True
        ).to(self.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(self.device)

        llm_tokens, input_part_targets_len = concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

        bs = inputs_embeds.shape[0]
        
        prompt = samples["prompt"]

        att_list = []
        inp_list = []
        # joint video audio TODO add seconds
        for pos in range(num['video']):
            if self.enumerate_inputs:
                enumeration_pos = self.llm_tokenizer(
                    [f"{'' if pos == 0 else ' '}({chr(97+pos)}) " for _ in prompt],
                    return_tensors="pt",
                    add_special_tokens=False if (pos!= 0) else True
                    ).to(self.device)
                enumeration_inputs_llm = self.llm_model.get_input_embeddings()(enumeration_pos.input_ids)
                enumeration_atts_llm = enumeration_pos.attention_mask.to(self.device)
                inp_list.extend([enumeration_inputs_llm])
                att_list.extend([enumeration_atts_llm])

            
            for modality in ['video', 'audio']:
                att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality].view(bs,  num[modality], self.num_query_token)[:, pos, :]])
                inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality].view(bs,  num[modality], self.num_query_token, -1)[:, pos, :, :]])
             
             
             
        
        
        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(torch.cat(att_list, dim=1).size(), dtype=torch.long).to(self.device).fill_(-100)
        )

        # append llm prompt + output to queries
        att_list.append(llm_tokens['attention_mask'])
        inp_list.append(inputs_embeds)
        
        inputs_embeds = torch.cat(inp_list, dim=1)
        attention_mask = torch.cat(att_list, dim=1)
        targets = torch.cat([empty_targets, targets], dim=1)

       
        
        with maybe_autocast(self):
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = dummy_loss+outputs.loss



        return {"loss": loss}