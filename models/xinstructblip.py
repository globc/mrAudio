from lavis.processors.audio_processors import BeatsAudioProcessor
from lavis.processors.alpro_processors import AlproVideoEvalProcessor
from omegaconf import OmegaConf
from lavis.common.registry import registry
from transformers import LlamaForCausalLM, LlamaTokenizer
from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
import torch
import random
import contextlib

def maybe_autocast(self, dtype=torch.float16):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    enable_autocast = self.device != torch.device("cpu")

    if enable_autocast:
        return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return contextlib.nullcontext()

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
        
        self.maybe_autocast = maybe_autocast

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
    

    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')
        if samples is None or samples == {} or not any([modality in samples for modality in self.modalities]):
            return {"loss": torch.tensor(0.0)}

        random.shuffle(self.modalities)

        a = LlamaForCausalLM.from_pretrained("./models/vicuna7b_v2.yaml")
        a.forward(samples)

        curr_modalities = [modality for modality in self.modalities if modality in samples]
        excess_modalities = [modality for modality in self.modalities if modality not in curr_modalities]
        # disable gradient in excess modalities
        dummy_loss = 0.
        for modality in excess_modalities:
            if self.shared_qformer:
                for name, param in getattr(self, f"{modality}_encoder_projection").named_parameters():
                    # param.requires_grad = False
                    dummy_loss += param.sum()*0.
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
            if modality == "video" :  ## assert eva_clip_g
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(2)):
                    this_frame = data[:,:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][-1] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                        data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                # B, Token Size, LM EMB
                if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                    query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(data.size(0), -1, -1)
                    
        query_outputs = {}
        if self.qformer_text_input:
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
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    num = len(embeds[modality])
                    bs = embeds[modality][0].shape[0]
                    indices = [j_+r for r,j in enumerate([[i*bs for i in range(num)]]*bs) for j_ in j]
                    reordered_embeds = torch.cat(embeds[modality])[indices]
                    reordered_atts = torch.cat(data_atts[modality])[indices]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num, self.num_query_token, -1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                        continue
                    query_output = getattr(self, f"{modality}_Qformer").bert(
                        text_Qformer.input_ids.repeat(num, 1),
                        attention_mask=Qformer_atts[modality].repeat(num, 1),
                        query_embeds=query_tokens[modality].repeat(num, 1, 1),
                        encoder_hidden_states=reordered_embeds,
                        encoder_attention_mask=reordered_atts,
                        return_dict=True,
                    )
                    query_outputs[modality] = query_output
                else:
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                        continue  
                    query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts[modality],
                        query_embeds=query_tokens[modality], 
                        encoder_hidden_states=embeds[modality].to(torch.float32), 
                        encoder_attention_mask=data_atts[modality], 
                        return_dict=True,
                    )
        else:
            for modality in curr_modalities:
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    num = len(embeds[modality])
                    bs  = embeds[modality][0].shape[0]
                    indices = [j_+r for r,j in enumerate([[i*bs for i in range(num)]]*bs) for j_ in j]
                    reordered_embeds = torch.cat(embeds[modality])[indices]
                    reordered_atts = torch.cat(data_atts[modality])[indices]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num, self.num_query_token, -1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                        continue
                    query_output = getattr(self, f"{modality}_Qformer").bert(
                        query_embeds=query_tokens[modality].repeat(num, 1, 1),
                        encoder_hidden_states=reordered_embeds,
                        encoder_attention_mask=reordered_atts,
                        return_dict=True,
                    )
                    query_outputs[modality] = query_output
                else: 
                    bs = embeds[modality].shape[0]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                        continue  
                    query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                        query_embeds=query_tokens[modality],
                        encoder_hidden_states=embeds[modality].to(torch.float32), # pc data is floa16.
                        encoder_attention_mask=data_atts[modality],
                        return_dict=True,
                    )
        
        inputs_llm = {}
        atts_llm = {}
        for modality in curr_modalities:
            if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                # num*bs, num query tokens, llm emb size
                if self.projection_only or getattr(self, f"projection_only_{modality}"):
                    if self.proj_dim != 1:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].unsqueeze(1)).reshape(bs*num, self.num_query_token, -1)
                    else:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality]).reshape(bs*num, self.num_query_token, -1)
                    inputs_llm[modality] = inputs_llm[modality].reshape(bs, num, self.num_query_token, -1).view(bs, num*self.num_query_token, -1)
                    atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                    continue
                inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:]) 
                # bs, num, num query tokens, llm emb size -> bs, num*num query tokens, llm emb size
                inputs_llm[modality] = inputs_llm[modality].reshape(bs, num, self.num_query_token, -1).view(bs, num*self.num_query_token, -1)
                atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
            else:
                if self.projection_only or getattr(self, f"projection_only_{modality}"):
                    if self.proj_dim == 1:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].mean(-1)).reshape(bs, self.num_query_token, -1)
                    else:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].reshape(bs, self.num_query_token, -1))
                    atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                    continue
                inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:])
                atts_llm[modality] = torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'

        if self.llm_text_input:
            text_input_tokens = self.llm_tokenizer(
                [f"{t}{self.postfix}" for t in samples['text_input']] if self.postfix else samples['text_input'],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens= not self.clean_tokenization
            ).to(self.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(self.device)

        if self.llm_text_input:
            llm_tokens, input_part_targets_len = self.concat_text_input_output(
                text_input_tokens.input_ids,
                text_input_tokens.attention_mask,
                text_output_tokens.input_ids,
                text_output_tokens.attention_mask,
            )
        else:
            llm_tokens = text_output_tokens
            input_part_targets_len = [0 for _ in range(llm_tokens['input_ids'].shape[0])] # input length is 0

        
        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

        bs = inputs_embeds.shape[0]

        att_list = []
        inp_list = []

        if self.prefix:
            att_list = [self.tokenized_prefix.attention_mask.repeat(bs, 1).to(self.device)]
            inp_list = [self.llm_model.get_input_embeddings()(self.tokenized_prefix.input_ids.to(self.device)).repeat(bs, 1, 1)]            
        for modality in curr_modalities:
            if self.use_cues:
                if self.prefix and self.clean_tokenization:
                     att_list.extend([self.att_cue[modality][:,1:].repeat(bs, 1).to(self.device), atts_llm[modality]])
                     inp_list.extend([self.emb_cue[modality][:,1:].repeat(bs, 1, 1).to(self.device), inputs_llm[modality]])
                att_list.extend([self.att_cue[modality].repeat(bs, 1).to(self.device), atts_llm[modality]])
                inp_list.extend([self.emb_cue[modality].repeat(bs, 1, 1).to(self.device), inputs_llm[modality]])
            else:
                att_list.extend([atts_llm[modality]])
                inp_list.extend([inputs_llm[modality]])
       
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

       
        
        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = dummy_loss+outputs.loss
        return {"loss": loss}

    def forward_new(self,samples):
        
        if samples == None or samples == {} or not any([modality in samples for modality in self.modalities]):
            return {"loss": torch.tensor(0.0)}

        random.shuffle(self.modalities)

        curr_modalities = [modality for modality in self.modalities if modality in samples]
        excess_modalities = [modality for modality in self.modalities if modality not in curr_modalities]
        # disable gradient in excess modalities
        dummy_loss = 0.
        for modality in excess_modalities:
            if self.shared_qformer:
                for name, param in getattr(self, f"{modality}_encoder_projection").named_parameters():
                    # param.requires_grad = False
                    dummy_loss += param.sum()*0.
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
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][-1] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                        data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                # B, Token Size, LM EMB
                query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(data.size(0), -1, -1)
                    
            elif modality == 'audio':
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(1)):
                    this_frame = data[:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][j] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                    data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
                # B, Token Size, LM EMB
                query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(data.size(0), -1, -1)
        

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
                    
        curr_modalities = [modality for modality in self.modalities if modality in samples]
        query_tokens = {}
        inputs_llm = {}
        atts_llm = {}
        query_outputs = {}
        for modality in curr_modalities:
                inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:]) 
                # bs, num, num query tokens, llm emb size -> bs, num*num query tokens, llm emb size
                inputs_llm[modality] = inputs_llm[modality].reshape(bs, num, self.num_query_token, -1).view(bs, num*self.num_query_token, -1)
                atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)   
        
        

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'

        text_input_tokens = self.llm_tokenizer(
            [f"{t}{self.postfix}" for t in samples['text_input']] if self.postfix else samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens= not self.clean_tokenization
        ).to(self.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(self.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
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
                    add_special_tokens=False if (pos!= 0 or self.prefix) else True
                    ).to(self.device)
                enumeration_inputs_llm = self.llm_model.get_input_embeddings()(enumeration_pos.input_ids)
                enumeration_atts_llm = enumeration_pos.attention_mask.to(self.device)
                inp_list.extend([enumeration_inputs_llm])
                att_list.extend([enumeration_atts_llm])

            
            for modality in ['video', 'audio']:
                if self.clean_tokenization:
                    if self.prefix or pos > 1 or self.enumerate_inputs or modality == 'audio':
                        att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask[:,1:]).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality].view(bs,  num[modality], self.num_query_token)[:, pos, :]])
                        inp_list.extend([self.emb_cue[modality][:,1:].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality].view(bs,  num[modality], self.num_query_token, -1)[:, pos, :, :]])
                        continue
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

       
        
        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = dummy_loss+outputs.loss



        return {"loss": loss}