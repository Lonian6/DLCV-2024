import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import loralib as lora
import copy
import numpy as np



class Config:
    def __init__(self, checkpoint=None, lora_ckpt=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        self.lora_ckpt = lora_ckpt
        self.rank = 32
        self.lora_dropout=0.2

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, cfg.rank)
        # self.c_attn = lora.MergedLinear(cfg.n_embd, 3 * cfg.n_embd, r=cfg.rank, 
        #                                 enable_lora=[True, False, True], 
        #                                 # lora_alpha=cfg.lora_attn_alpha, 
        #                                 lora_dropout=cfg.lora_dropout,)
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, cfg.rank)
        # self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        # self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        query = q
        key = k
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # query = q
        # key = k
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C)), query, key

class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        # self.mlp = nn.Sequential(collections.OrderedDict([
        #     ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
        #     ('act', nn.GELU(approximate='tanh')),
        #     ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        # ]))
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, cfg.rank)), 
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, cfg.rank)) 
        ]))

    def forward(self, x):
        out, _, _ = self.attn(self.ln_1(x))
        x = x + out
        x = x + self.mlp(self.ln_2(x))
        return x

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            # wte = lora.Embedding(cfg.vocab_size, cfg.n_embd, cfg.rank),
            # wpe = lora.Embedding(cfg.block_size, cfg.n_embd, cfg.rank),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        # self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, cfg.rank, bias=False)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor, img_embeddings: Tensor):
        # v2
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        # x = self.transformer.wte(x) + self.transformer.wpe(pos)
        x = self.transformer.wte(x)
        x = torch.cat((img_embeddings, x), dim=1)
        
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        pos = self.transformer.wpe(pos)
        x = x + pos
        
        x = self.lm_head(self.transformer.ln_f(self.transformer.h(x)))
        return x

def temperature_sampling(logits, temperature, topk):
    # logits=[B, L, Vocab]
    # logits = torch.Tensor(logits)
    probs = nn.Softmax(dim=0)(logits / temperature)
    probs = np.array(probs)
    if topk == 1:
        prediction = np.argmax(probs)
    else:
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:topk]
        candi_probs = [probs[i] for i in candi_index]
        # normalize probs
        candi_probs /= sum(candi_probs)
        # choose by predicted probs
        prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return prediction

class Caption_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.PAD_TOKEN = 50256
        self.UNK_TOKEN = 1
        self.BOS_TOKEN = 50256
        self.EOS_TOKEN = 50256
        
        self.decoder = Decoder(cfg)
        self.img_linear = nn.Linear(1024, 768)
        
        # self.img_linear = nn.Sequential(
        #     nn.LayerNorm(1024),
        #     nn.Linear(1024, 768),
        #     nn.GELU(),
        #     nn.LayerNorm(768),
        #     nn.Linear(768, 768),
        #     # nn.GELU(),
        #     # nn.LayerNorm(768),
        #     # nn.Linear(768, 768)
        # )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, img_embedding, caption, caption_mask=None):
        B, img_L, _ = img_embedding.shape
        B, caption_L = caption.shape
        
        img_embedding = self.img_linear(img_embedding) # [B, L, 768]
        decoder_output = self.decoder(caption[:, :-1], img_embedding)
        
        # decoder_input = torch.cat((img_embedding[:, :, 0], caption[:, :-1]), dim=1)
        # print()
        target = torch.full((B, img_L+caption_L-1), fill_value=-100)
        caption_m = copy.deepcopy(caption)
        if caption_mask is not None:
            caption_m[caption_mask == 0] = -100
        
        decoder_output = decoder_output.permute(0, 2, 1)
        
        if self.training:
            target[:, -caption_L:] = caption_m
            loss = self.criterion(decoder_output, target.to(decoder_output.device).long())
        else:
            loss = None
        
        return decoder_output, loss
    
    def pred_forward(self, img_embedding, caption):
        _, caption_L = caption.shape
        img_embedding = self.img_linear(img_embedding) # [B, L, 768]
        decoder_output = self.decoder(caption, img_embedding)
        # decoder_output = decoder_output.permute(0, 2, 1)
        return decoder_output
    
    def inference(self, img_embedding, temperature=1.2, topk=100, max_length=20):
        self.eval()
        batch = img_embedding.shape[0]
        
        device = img_embedding.device
        current_caption = torch.tensor([50256]).to(device).unsqueeze(1).repeat(batch, 1)
        
        for i in range(max_length - 1):
            # print('================================================================')
            # print(current_caption.shape)
            # a = input('========================')
            logits = self.pred_forward(img_embedding, current_caption)
            # print(logits.shape)
            # a = input('========================')
            word = []
            for b in range(batch):
                # if current_caption[b, -1] == 50256 and len(current_caption[b, :]) != 1:
                #     word.append(50256)
                # else:
                last_logits = logits[b, -1, :].to('cpu').detach()
                next_chars = temperature_sampling(last_logits, temperature, topk)
                word.append(next_chars)
                # print(next_chars)
                # a = input('========================')
            # print(np.array(word).shape)
            current_caption = torch.cat((current_caption, torch.Tensor(word).view(batch, 1).to(device).long()), axis=1)
        current_caption = torch.cat((current_caption, torch.tensor([50256]).to(device).unsqueeze(1).repeat(batch, 1)), axis=1)
        return current_caption
    
    def beam_search(self, img_embedding, beams=3, max_length=30):

        self.eval()
        device = img_embedding.device
        current_state = torch.tensor([50256]).to(device).unsqueeze(1)
        caption_length = current_state.shape[1]

        # get top k words
        next_probs = self.pred_forward(img_embedding, current_state) # [B, L, vocab]
        # print(next_probs.shape)
        next_probs = next_probs[:, -1]
        vocab_size = next_probs.shape[-1] # 50257
        current_probs, next_chars = next_probs.log_softmax(-1).topk(k=beams, axis=-1)
        # print('\n===========current_probs, next_chars=======================')
        # print(current_probs, current_probs.shape)   # torch.Size([1, 1, 3])
        # print(next_chars, next_chars.shape)         # torch.Size([1, 1, 3])
        # a = input('==================================')
        # [1, 3]
        current_probs = current_probs.reshape(beams)    
        next_chars = next_chars.reshape(beams, 1)       
        # print('\n==============re====================')
        # print(current_probs, current_probs.shape)   # torch.Size([3])
        # print(next_chars, next_chars.shape)         # torch.Size([3, 1])
        # a = input('==================================')

        # gen first k beams
        # img_embedding = img_embedding.repeat((beams, 1, 1))
        current_state = current_state.repeat((beams, 1))
        current_state = torch.cat((current_state, next_chars), axis=1)
        
        # print('\n==============current_probs, img_embedding, current_state====================')
        # print(current_probs.shape) # torch.Size([3])
        # print(img_embedding.shape) # torch.Size([3, 257, 1024])
        # print(current_state.shape) # torch.Size([3, 2])
        # a = input('==================================')
        
        ans_ids = []
        ans_probs = []
        for i in range(max_length - 1):
            caption_length = current_state.shape[1]
            img_embeddings = img_embedding.repeat((beams, 1, 1))
            # get top k beams for beam*beam candidates
            # print('\n==============current_probs, img_embedding, current_state====================')
            # print(current_probs.shape) # torch.Size([3])
            # print(img_embeddings.shape) # torch.Size([3, 257, 1024])
            # print(current_state.shape) # torch.Size([3, 2])
            # a = input('==================================')
            next_probs = self.pred_forward(img_embeddings, current_state).log_softmax(-1)
            next_probs = next_probs[:, -1]
            # print('\n==============re====================')
            # print(next_probs.shape) # torch.Size([3, 1, 50257])
            # a = input('==================================')
            
            current_probs = current_probs.unsqueeze(-1) + next_probs
            current_probs = current_probs.flatten()  # (beams*vocab)

            # length normalization
            _, idx = (current_probs / (len(current_state[0]) + 1))\
                .topk(k=beams, dim=-1)
            current_probs = current_probs[idx]

            # get corresponding next char
            next_chars = torch.remainder(idx, vocab_size)
            next_chars = next_chars.unsqueeze(-1)

            # get corresponding original beams
            top_candidates = (idx / vocab_size).long()
            current_state = current_state[top_candidates]
            current_state = torch.cat((current_state, next_chars), dim=1)

            # concat next_char to beams
            to_rm_idx = set()
            for idx, ch in enumerate(next_chars):
                if i == (max_length - 2) or ch.item() == 50256:
                    ans_ids.append(current_state[idx].cpu().tolist())
                    ans_probs.append(
                        current_probs[idx].item() / len(ans_ids[-1]))
                    to_rm_idx.add(idx)
                    beams -= 1
            to_keep_idx = [i for i in range(
                len(current_state)) if i not in to_rm_idx]
            if len(to_keep_idx) == 0:
                break
            current_state = current_state[to_keep_idx]
            current_probs = current_probs[to_keep_idx]

        max_idx = torch.argmax(torch.tensor(ans_probs)).item()
        return ans_ids[max_idx]