from os import name
from pathlib import Path
from tempfile import TemporaryDirectory
from header import *
from huggingface_hub import HfApi, HfFolder

class Copyisallyouneed(nn.Module):

    def __init__(self, **args):
        super(Copyisallyouneed, self).__init__()
        self.args = args

        # bert-encoder model
        self.phrase_encoder = AutoModel.from_pretrained(
            self.args['phrase_encoder_model'][self.args['lang']]
        )
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            self.args['phrase_encoder_tokenizer'][self.args['lang']]
        )
        self.bert_tokenizer.add_tokens(['<|endoftext|>', '[PREFIX]'])
        self.prefix_token_id = self.bert_tokenizer.convert_tokens_to_ids('[PREFIX]')
        self.phrase_encoder.resize_token_embeddings(self.phrase_encoder.config.vocab_size+2)
        
        # model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args['prefix_encoder_tokenizer'][self.args['lang']])
        self.vocab_size = len(self.tokenizer)
        self.pad = self.tokenizer.pad_token_id if self.args['lang'] == 'zh' else self.tokenizer.bos_token_id

        self.model = GPT2LMHeadModel.from_pretrained(self.args['prefix_encoder_model'][self.args['lang']])
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        # MLP: mapping bert phrase start representations
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)
        )
        # MLP: mapping bert phrase end representations
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)
        )
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return output

    def get_token_loss(self, ids, hs):
        # no pad token
        label = ids[:, 1:]
        logits = torch.matmul(
            hs[:, :-1, :],
            self.token_embeddings.t()
        )
        # TODO: inner loss function remove the temperature factor
        logits /= self.args['temp']
        loss = self.gen_loss_fct(logits.view(-1, logits.size(-1)), label.reshape(-1))
        chosen_tokens = torch.max(logits, dim=-1)[1]
        gen_acc = (chosen_tokens.reshape(-1) == label.reshape(-1)).to(torch.long)
        valid_mask = (label != self.pad).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc

    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['gpt2_ids'], batch['gpt2_mask']
        last_hidden_states = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True).hidden_states[-1]
        # get token loss
        loss_0, acc_0 = self.get_token_loss(ids, last_hidden_states)

        ## encode the document with the BERT encoder model
        dids, dids_mask = batch['bert_ids'], batch['bert_mask']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        # collect the phrase start representations and phrase end representations
        s_rep = self.s_proj(output)
        e_rep = self.e_proj(output)    
        s_rep = s_rep.reshape(-1, s_rep.size(-1))
        e_rep = e_rep.reshape(-1, e_rep.size(-1))    # [B_doc*S_doc, 768//2]

        # collect the query representations
        query = last_hidden_states[:, :-1].reshape(-1, last_hidden_states.size(-1))
        query_start = query[:, :self.model.config.hidden_size//2]
        query_end = query[:, self.model.config.hidden_size//2:]

        # training the representations of the start tokens
        candidate_reps = torch.cat([
            self.token_embeddings[:, :self.model.config.hidden_size//2], 
            s_rep], dim=0)
        logits = torch.matmul(query_start, candidate_reps.t())
        logits /= self.args['temp']

        # build the padding mask for query side
        query_padding_mask = ids_mask[:, :-1].reshape(-1).to(torch.bool)
        
        # build the padding mask: 1 for valid and 0 for mask
        attention_mask = (dids_mask.reshape(1, -1).to(torch.bool)).to(torch.long)
        padding_mask = torch.ones_like(logits).to(torch.long)
        # Santiy check over
        padding_mask[:, self.vocab_size:] = attention_mask

        # build the position mask: 1 for valid and 0 for mask
        pos_mask = batch['pos_mask']
        start_labels, end_labels = batch['start_labels'][:, 1:].reshape(-1), batch['end_labels'][:, 1:].reshape(-1)
        position_mask = torch.ones_like(logits).to(torch.long)
        query_pos = start_labels > self.vocab_size
        # ignore the padding mask
        position_mask[query_pos, self.vocab_size:] = pos_mask
        assert padding_mask.shape == position_mask.shape
        # overall mask
        overall_mask = padding_mask * position_mask
        ## remove the position mask
        # overall_mask = padding_mask

        new_logits = torch.where(overall_mask.to(torch.bool), logits, torch.tensor(-1e4).to(torch.half).cuda())
        mask = torch.zeros_like(new_logits)
        mask[range(len(new_logits)), start_labels] = 1.
        loss_ = F.log_softmax(new_logits[query_padding_mask], dim=-1) * mask[query_padding_mask]
        loss_1 = (-loss_.sum(dim=-1)).mean()

        ## split the token accuaracy and phrase accuracy
        phrase_indexes = start_labels > self.vocab_size
        phrase_indexes_ = phrase_indexes & query_padding_mask
        phrase_start_acc = new_logits[phrase_indexes_].max(dim=-1)[1] == start_labels[phrase_indexes_]
        phrase_start_acc = phrase_start_acc.to(torch.float).mean().item()
        phrase_indexes_ = ~phrase_indexes & query_padding_mask
        token_start_acc = new_logits[phrase_indexes_].max(dim=-1)[1] == start_labels[phrase_indexes_]
        token_start_acc = token_start_acc.to(torch.float).mean().item()

        # training the representations of the end tokens
        candidate_reps = torch.cat([
            self.token_embeddings[:, self.model.config.hidden_size//2:], 
            e_rep], dim=0
        )
        logits = torch.matmul(query_end, candidate_reps.t())    # [Q, B*]  
        logits /= self.args['temp']
        new_logits = torch.where(overall_mask.to(torch.bool), logits, torch.tensor(-1e4).to(torch.half).cuda())
        mask = torch.zeros_like(new_logits)
        mask[range(len(new_logits)), end_labels] = 1.
        loss_ = F.log_softmax(new_logits[query_padding_mask], dim=-1) * mask[query_padding_mask]
        loss_2 = (-loss_.sum(dim=-1)).mean()
        # split the phrase and token accuracy
        phrase_indexes = end_labels > self.vocab_size
        phrase_indexes_ = phrase_indexes & query_padding_mask
        phrase_end_acc = new_logits[phrase_indexes_].max(dim=-1)[1] == end_labels[phrase_indexes_]
        phrase_end_acc = phrase_end_acc.to(torch.float).mean().item()
        phrase_indexes_ = ~phrase_indexes & query_padding_mask
        token_end_acc = new_logits[phrase_indexes_].max(dim=-1)[1] == end_labels[phrase_indexes_]
        token_end_acc = token_end_acc.to(torch.float).mean().item()
        return (
            loss_0,     # token loss
            loss_1,     # token-head loss
            loss_2,     # token-tail loss
            acc_0,      # token accuracy
            phrase_start_acc,
            phrase_end_acc,
            token_start_acc,
            token_end_acc
        )

    # def save_pretrained(self, save_directory):
    #     print(f"[!] saving model to {save_directory}")
    #     if not os.path.exists(save_directory):
    #         os.makedirs(save_directory)
    #     torch.save(self.state_dict(), f"{save_directory}/model.safetensors")
    #     with open(f"{save_directory}/config.json", "w") as f:
    #         json.dump(self.args, f)
    #     self.tokenizer.save_pretrained(save_directory)
    #     self.bert_tokenizer.save_pretrained(save_directory)
    #     self.push_to_hub(save_directory)
        
    
    def save_pretrained(self, save_directory):
        print(f"[!] saving model to {save_directory}")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the encoder and decoder configurations using Hugging Face's built-in functionality
        encoder_save_dir = os.path.join(save_directory, "encoder")
        decoder_save_dir = os.path.join(save_directory, "decoder")

        # Save the encoder (BERT) configuration and model
        self.phrase_encoder.save_pretrained(encoder_save_dir)
        self.bert_tokenizer.save_pretrained(encoder_save_dir)

        # Save the decoder (GPT-2) configuration and model
        self.model.save_pretrained(decoder_save_dir)
        self.tokenizer.save_pretrained(decoder_save_dir)

        # Save the overall configuration if you want to combine them in a central config.json
        encoder_config = json.load(open(f"{encoder_save_dir}/config.json", "r"))
        decoder_config = json.load(open(f"{decoder_save_dir}/config.json", "r"))
        
        config = {
            "model_type": "encoder-decoder",
            "encoder": encoder_config,
            "decoder": decoder_config,
            "dropout": self.args['dropout'],
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "temperature": self.args['temp'],
            "lang": self.args['lang']
        }

        # Save the overall config.json
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)
        print(f"[!] model saved to {save_directory}")
        self.push_to_hub(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory):
        with open(f"{load_directory}/config.json", "r") as f:
            args = json.load(f)
        model = cls(**args)

        # Load the encoder and decoder models using their respective directories
        encoder_save_dir = os.path.join(load_directory, "encoder")
        decoder_save_dir = os.path.join(load_directory, "decoder")

        model.phrase_encoder = AutoModel.from_pretrained(encoder_save_dir)
        model.bert_tokenizer = AutoTokenizer.from_pretrained(encoder_save_dir)

        model.model = GPT2LMHeadModel.from_pretrained(decoder_save_dir)
        model.tokenizer = AutoTokenizer.from_pretrained(decoder_save_dir)

        return model

    
    def push_to_hub(self, directory):
        repo_name = self.args['hf_model_name']
        api = HfApi()
        token = HfFolder.get_token()
        print(f"[!] pushing model to the hub: {repo_name}")
        api.create_repo(
            repo_id=repo_name,
            private=True,
            token=token,
            exist_ok=True
        )
        print(f"[!] uploading model to the hub: {repo_name}")
        api.upload_folder(
            folder_path=directory,
            path_in_repo="",
            repo_id=repo_name,
            token=token
        )