!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers datasets wandb tqdm torchinfo
!pip install liger-kernel
!pip install accelerate

# Login to Weights & Biases 
!wandb login

# Login to Hugging Face (used this for dataset access)
!huggingface-cli login

#main

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import torch.optim as optim
from transformers.models.albert import AlbertTokenizer
from datasets import load_dataset
import os

# Set random seeds for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

#config

TOKEN = os.environ.get('HF_TOKEN', '')  #hugging face token goes here, removed it to push to git

# Hyperparams
beta_2 = 0.98
eps = 1e-9
beta_1 = 0.9
block_size = 512
batch_size = 32
embeddings_dims = 512
no_of_heads = 8
dropout = 0.1
epochs = 1
max_lr = 6e-4
no_of_decoder_layers = 6
attn_dropout = 0.1
weight_decay_optim = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip = 1.0
use_liger = True

# Training parameters
save_checkpoint_iter = 500
total_iters = 10000
eval_iters = 200
eval_check = 200
warmup_iters = 400
min_lr = 0.1 * max_lr
lr_decay_iters = 10000
total_batch_size = 524288
micro_batch_size = batch_size
gradient_accumulation_steps = total_batch_size // (micro_batch_size * block_size)

print(f"Using device: {device}")
print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

#load data

def load_datasets(token, sample_size=None):
    train_dataset = load_dataset("ai4bharat/samanantar", 'hi', split="train", token=token)
    print("dataset loaded")
    return train_dataset

fw_train = load_datasets(TOKEN)
fw_train = fw_train.train_test_split(test_size=0.01)
print(f"Train size: {len(fw_train['train'])}, Val size: {len(fw_train['test'])}")

# Load tokenizer
tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/IndicBARTSS")
src_vocab_size = len(tokenizer)
tgt_vocab_size = len(tokenizer)

#utility
def _save_snapshot(model, optimizer, scheduler, epoch, step):
    snapshot = {
        "MODEL_STATE": model.state_dict(),
        "OPTIMIZER_STATE": optimizer.state_dict(),
        "EPOCHS_RUN": epoch,
        "STEP_RUN": step
    }

  
    os.makedirs('/content/checkpoints', exist_ok=True)
    torch.save(snapshot, f"/content/checkpoints/snapshot_{step}.pt")
    print(f"Step: {step} | Snapshot saved to /content/checkpoints/snapshot_{step}.pt")

def prepare_dataset(split, device, batch_size):
    print(f"Prepping {split} dataset")
    
    def collate_fn(batch):
        en_texts = []
        hi_texts = []
        
        for item in batch:
            it = tokenizer.bos_token + item['src'] + tokenizer.eos_token 
            en_texts.append(it)
            it = item['tgt'] + tokenizer.eos_token
            hi_texts.append(it)

        input_encodings = tokenizer(en_texts, padding='max_length', max_length=block_size, 
                                    truncation=True, add_special_tokens=False, return_tensors="pt")
        target_encodings = tokenizer(hi_texts, padding='max_length', max_length=block_size, 
                                     truncation=True, return_tensors="pt", add_special_tokens=False)
        
        input_encodings["labels"] = target_encodings["input_ids"].clone()
        input_encodings["decoder_input_ids"] = target_encodings["input_ids"].clone()
        input_encodings["decoder_input_ids"][:, 1:] = input_encodings["decoder_input_ids"][:, :-1]
        input_encodings["decoder_input_ids"][:, 0] = tokenizer.bos_token_id
        
        return input_encodings

    if split == 'train':
        data_loader = DataLoader(
            fw_train['train'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True,
        )
    elif split == 'val':
        data_loader = DataLoader(
            fw_train['test'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False,
        )
    return data_loader

def find_unused_parameters(model):
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused

def get_lr(it):
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (max_lr - min_lr)

#model components, essential ones

class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, embeddings_dims=embeddings_dims, max_seq_len=block_size, theta=10000.0):
        super().__init__()
        self.embeddings_dims = embeddings_dims
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        pe = torch.zeros(max_seq_len, embeddings_dims)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embeddings_dims, 2).float() * 
                           -(math.log(theta) / embeddings_dims))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.shape[1]
        pe = getattr(self, 'pe')
        return pe[:, :seq_len, :]

class TgtTextEmbeddings(nn.Module):
    def __init__(self, vocab_size=tgt_vocab_size, embeddings_dims=embeddings_dims):
        super().__init__()
        self.embeddings_table = nn.Embedding(num_embeddings=tgt_vocab_size, 
                                             embedding_dim=embeddings_dims, device=device)

    def forward(self, x):
        return self.embeddings_table(x)

class SrcTextEmbeddings(nn.Module):
    def __init__(self, vocab_size=src_vocab_size, embeddings_dims=embeddings_dims):
        super().__init__()
        self.embeddings_table = nn.Embedding(num_embeddings=src_vocab_size, 
                                             embedding_dim=embeddings_dims, device=device)

    def forward(self, x):
        return self.embeddings_table(x)

class LayerNormalization(nn.Module):
    def __init__(self, embeddings_dims=embeddings_dims):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=embeddings_dims)
    
    def forward(self, x):
        return self.norm(x)

class MLPBlock(nn.Module):
    def __init__(self, dropout=dropout, embeddings_size=embeddings_dims):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(device=device, in_features=embeddings_size, out_features=4 * embeddings_dims),
            nn.GELU(),
            nn.Linear(device=device, in_features=4 * embeddings_dims, out_features=embeddings_size),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.mlp(x)

class MaskedAttentionHead(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, x, mask=None):
        batch, block_size, embd_dims = x.shape
        k = self.keys(x)
        q = self.query(x)
        v = self.values(x)
        weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            masked_values = weights.masked_fill(mask == 0, float('-inf'))
            masked_table = torch.tril(torch.ones(block_size, block_size, device=device))
            masked_values = masked_values.masked_fill(masked_table[:block_size, :block_size] == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1)
            out = weights_normalized @ v
            return out
        else:
            masked_table = torch.tril(torch.ones(block_size, block_size, device=device))
            weights = weights.masked_fill(masked_table[:block_size, :block_size] == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(weights, dim=-1)
            out = weights_normalized @ v
            return out

class MaskedMHA(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads):
        super().__init__()
        self.heads = nn.ModuleList([MaskedAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, 
                                                        no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p=attn_dropout)
        self.linear = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=device, bias=False)

    def forward(self, x, mask=None):
        concat = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out

class CrossAttentionHead(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, k, v, q, srcmask=None):
        query = self.query(q)
        key = self.keys(k)
        value = self.values(v)
        weights = query @ torch.transpose(key, dim0=-2, dim1=-1) * (key.shape[-1] ** -0.5)

        if srcmask is not None:
            srcmask = srcmask.unsqueeze(1)
            masked_values = weights.masked_fill(srcmask == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1)
            out = weights_normalized @ value
            return out
        else:
            weights_normalized = nn.functional.softmax(weights, dim=-1)
            out = weights_normalized @ value
            return out

class FullAttentionHead(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, x, mask=None):
        k = self.keys(x)
        q = self.query(x)
        v = self.values(x)
        weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            masked_values = weights.masked_fill(mask == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1)
            out = weights_normalized @ v
            return out
        else:
            weights_normalized = nn.functional.softmax(weights, dim=-1)
            weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ v
            out = self.dropout(out)
        return out

class FullMHA(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads):
        super().__init__()
        self.heads = nn.ModuleList([FullAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, 
                                                      no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p=attn_dropout)
        self.linear = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=device, bias=False)

    def forward(self, x, mask=None):
        concat = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out

class CrossMHA(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, 
                                                       no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p=attn_dropout)
        self.linear = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=device, bias=False)

    def forward(self, value, key, x, srcmask=None):
        concat = torch.cat([head(value, key, x, srcmask) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out

class TransformerDecoderBlock(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, 
                 no_of_heads=no_of_heads, dropout=dropout):
        super().__init__()
        self.cross = CrossMHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads)
        self.masked = MaskedMHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads)
        self.layer_norm1 = LayerNormalization(embeddings_dims)
        self.layer_norm2 = LayerNormalization(embeddings_dims)
        self.layer_norm4 = LayerNormalization(embeddings_dims)
        self.mlp_block = MLPBlock(dropout=dropout, embeddings_size=embeddings_dims)

    def forward(self, key, value, x, Srcmask=None, Targetmask=None):
        x = self.layer_norm1(x + self.masked(x, Targetmask))
        x = self.layer_norm2(x + self.cross(key, value, x, Srcmask))
        x = self.layer_norm4(x + self.mlp_block(x))
        return x

class DecoderModel(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, 
                 no_of_heads=no_of_heads, block_size=block_size, dropout=dropout, 
                 no_of_decoder_layers=no_of_decoder_layers):
        super().__init__()
        self.tgt_text_embds = TgtTextEmbeddings(vocab_size=tgt_vocab_size, embeddings_dims=embeddings_dims)
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(attn_dropout=attn_dropout, 
                                                                     embeddings_dims=embeddings_dims, 
                                                                     no_of_heads=no_of_heads, 
                                                                     dropout=dropout) for _ in range(no_of_decoder_layers)])
        self.apply(self._init_weights)
        self.positional_embeddings_tgt = SinusoidalPositionalEmbeddings(
            embeddings_dims=embeddings_dims, 
            max_seq_len=block_size, 
            theta=10000.0
        )
        self.dropout = nn.Dropout(p=dropout)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, key, value, x, srcmask=None, target_mask=None):
        x = self.tgt_text_embds(x)
        x = self.dropout(x)
        x = x + self.positional_embeddings_tgt(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(key, value, x, srcmask, target_mask)
        x = self.dropout(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, 
                 no_of_heads=no_of_heads, dropout=dropout, mask=None):
        super().__init__()
        self.mha = FullMHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads)
        self.layer_norm1 = LayerNormalization(embeddings_dims)
        self.layer_norm2 = LayerNormalization(embeddings_dims)
        self.mlp_block = MLPBlock(dropout=dropout, embeddings_size=embeddings_dims)

    def forward(self, x, mask=None):
        x = self.layer_norm1(x + self.mha(x, mask))
        x = self.layer_norm2(x + self.mlp_block(x))
        return x

class EncoderModel(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, 
                 no_of_heads=no_of_heads, block_size=block_size, dropout=dropout, 
                 no_of_decoder_layers=no_of_decoder_layers):
        super().__init__()
        self.positional_embeddings_src = SinusoidalPositionalEmbeddings(
            embeddings_dims=embeddings_dims, 
            max_seq_len=block_size, 
            theta=10000.0
        )
        self.src_text_embeds = SrcTextEmbeddings(vocab_size=src_vocab_size, embeddings_dims=embeddings_dims)
        self.encoder_layers = nn.ModuleList([TransformerEncoderBlock(attn_dropout=attn_dropout, 
                                                                     embeddings_dims=embeddings_dims, 
                                                                     no_of_heads=no_of_heads, 
                                                                     dropout=dropout) for _ in range(no_of_decoder_layers)])
        self.apply(self._init_weights)
        self.dropout = nn.Dropout(p=dropout)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask):
        x = self.src_text_embeds(x)
        x = x + self.positional_embeddings_src(x)
        x = self.dropout(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        x = self.dropout(x)
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()
        self.norm = LayerNormalization(embeddings_dims=embeddings_dims)
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=tgt_vocab_size, 
                                      device=device, bias=False)
        if use_liger:
            self.le_loss = LigerFusedLinearCrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, src, tgt_idx, tgt, src_mask=None, tgt_mask=None, inference=False):
        x = self.encoder(src, src_mask)
        x = self.decoder(x, x, tgt_idx, src_mask, tgt_mask)
        x = self.norm(x)
        
        if inference:
            out = self.linear_layer(x)
            return out
            
        if use_liger:  
            y = x.contiguous().view(-1, embeddings_dims)
            if tgt is not None:
                labels = tgt.contiguous().view(-1)
                loss = self.le_loss(self.linear_layer.weight, y, labels)
                return loss
        else:
            batch_size, seq_len, _ = x.shape
            logits = self.linear_layer(x)
            logits = logits.view(-1, tgt_vocab_size)
            targets = tgt.contiguous().view(-1)
            loss = F.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
            return loss

#generation fxns

def topk_sampling(model, prompt, device, max_length=30, top_k=50, temperature=0.8, repetition_penalty=1.2):
    """Generate text using top-k sampling"""
    model.eval()
    
    input_ids = tokenizer(prompt, add_special_tokens=True, max_length=max_length, 
                         padding='max_length', return_tensors="pt")['input_ids'].to(device)
    target_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    
    src_mask = torch.ones(input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype).to(device)
    src_mask = src_mask.masked_fill(input_ids == tokenizer.pad_token_id, 0)
    
    with torch.no_grad():
        encoder_output = model.encoder(input_ids, src_mask)
    
    for i in range(max_length):
        with torch.no_grad():
            decoder_output = model.decoder(encoder_output, encoder_output, target_ids, src_mask, None)
            logits = model.linear_layer(model.norm(decoder_output))
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            next_token = torch.multinomial(top_k_probs, num_samples=1)
            xcol = torch.gather(top_k_indices, -1, next_token)
            target_ids = torch.cat([target_ids, xcol], dim=1)
            
            if xcol.item() == tokenizer.eos_token_id:
                break
    
    model.train()
    generated_text = tokenizer.decode(target_ids[0])
    return generated_text

def beam_search_corrected(model, prompt, tokenizer, device, block_size, beam_width=5, 
                          max_length=50, temperature=1.0):
    """Generate text using beam search"""
    model.eval()
    model = model.to(device)

    inputs = tokenizer(prompt, add_special_tokens=True, max_length=block_size,
                      padding='max_length', truncation=True, return_tensors="pt")
    src_input_ids = inputs["input_ids"].to(device)
    src_mask = torch.ones(src_input_ids.shape[0], src_input_ids.shape[1], 
                         dtype=torch.long, device=device)
    src_mask = src_mask.masked_fill(src_input_ids == tokenizer.pad_token_id, 0)

    with torch.no_grad():
        encoder_output = model.encoder(src_input_ids, src_mask)

    initial_sequence = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
    beams = [(initial_sequence, 0.0)]
    completed_sequences = []

    for step in range(max_length):
        if not beams:
            break

        all_next_candidates = []

        for current_seq_tensor, current_score in beams:
            if current_seq_tensor[0, -1].item() == tokenizer.eos_token_id:
                completed_sequences.append((current_seq_tensor, current_score))
                continue

            decoder_input_ids = current_seq_tensor
            tgt_self_attention_mask = torch.ones_like(decoder_input_ids, device=device)

            with torch.no_grad():
                decoder_out = model.decoder(encoder_output, encoder_output, 
                                          decoder_input_ids, src_mask)
                logits_last_token = model.linear_layer(model.norm(decoder_out[:, -1, :]))
                logits_last_token = logits_last_token / temperature
                log_probs = F.log_softmax(logits_last_token, dim=-1)

            top_k_log_probs, top_k_indices = torch.topk(log_probs.squeeze(0), beam_width, dim=-1)

            for i in range(beam_width):
                next_token_id = top_k_indices[i].item()
                next_token_log_prob = top_k_log_probs[i].item()
                new_seq_tensor = torch.cat([current_seq_tensor, 
                                          torch.tensor([[next_token_id]], device=device)], dim=1)
                new_score = current_score + next_token_log_prob
                all_next_candidates.append((new_seq_tensor, new_score))

        new_beams = []
        temp_completed = []
        all_next_candidates.sort(key=lambda x: x[1], reverse=True)

        for cand_seq, cand_score in all_next_candidates:
            if cand_seq[0, -1].item() == tokenizer.eos_token_id:
                temp_completed.append((cand_seq, cand_score))
            elif len(new_beams) < beam_width:
                new_beams.append((cand_seq, cand_score))

        beams = new_beams
        completed_sequences.extend(temp_completed)

        if len(completed_sequences) >= beam_width:
            completed_sequences.sort(key=lambda x: x[1], reverse=True)
            if not beams or (beams and completed_sequences[beam_width-1][1] > beams[0][1]):
                break

    if not completed_sequences and beams:
        completed_sequences.extend(beams)
    elif not completed_sequences and not beams:
        model.train()
        return "[Beam Search Error: No sequences generated]"

    completed_sequences.sort(key=lambda x: x[1], reverse=True)

    if not completed_sequences:
        model.train()
        return "[Beam Search Error: No completed sequences found]"

    best_sequence_tensor = completed_sequences[0][0].squeeze(0)
    model.train()
    generated_text = tokenizer.decode(best_sequence_tensor)
    return generated_text

#train fxns

@torch.inference_mode()
def estimate_loss(val_loader, val_iterator, device):
    """Estimate validation loss"""
    out = {}
    epoch_losses = []
    
    print("Starting validation evaluation...")
    for step in range(eval_check):  
        try:
            batch = next(val_iterator)
        except StopIteration:
            val_loader_iterator = iter(val_loader)
            batch = next(val_loader_iterator)
        
        total_loss = 0  
        total_batches = 0 
        
        idx = batch['input_ids'].to(device)
        targets_idx = batch['decoder_input_ids'].to(device)
        targets = batch['labels'].to(device)
        
        src_mask = torch.ones(batch_size, block_size, dtype=idx.dtype).to(device)
        tgt_mask = torch.ones(batch_size, block_size, dtype=idx.dtype).to(device)
        
        src_mask = src_mask.masked_fill(idx == tokenizer.pad_token_id, 0)
        tgt_mask = tgt_mask.masked_fill(targets_idx == tokenizer.pad_token_id, 0)
        
        with torch.autocast(device_type=device, dtype=torch.float16):
            loss = model(idx, targets_idx, targets, src_mask, tgt_mask)
        
        total_loss += loss.item()
        total_batches += 1
        
    epoch_loss = total_loss / total_batches if total_batches > 0 else 0.0
    epoch_losses.append(epoch_loss)
    
    out['val'] = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    return out

def train():
    wandb.init(
        project='Translation',
    )
    print("Wandb initialized")
    model = Transformer()
    print(f"Model initialized on device: {device}")
  
    optimizer = optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(beta_1, beta_2),
        eps=eps,
    )
   
    #Compile model 
    try:
        model = torch.compile(model)
        print("Model compiled successfully")
    except:
        print("Model compilation not available, no compilation here")

    model = model.to(device)
    model.train()

    #data loaders
    train_dataloader = prepare_dataset('train', device, batch_size)
    val_loader = prepare_dataset('val', device, batch_size)
    print("Data loaders ready")

    train_data_iterator = iter(train_dataloader)
    val_data_iterator = iter(val_loader)
    token_count = 0

    os.makedirs('/content/generations', exist_ok=True)
    scaler = torch.amp.GradScaler(enabled=True)
    
    batches_per_epoch = len(train_dataloader)
    total_batches = batches_per_epoch * epochs
    print(f"Training info: {epochs} epochs, {batches_per_epoch} batches per epoch, {total_batches} total batches")

    for step in tqdm(range(total_iters), desc="Training"):
        if (step % eval_iters == 0) or step == total_iters - 1:
            model.eval()
            losses = estimate_loss(val_loader, val_data_iterator, device)
            model.train()

            avg_val_loss = losses['val']
            perplexity = torch.exp(torch.tensor(avg_val_loss))

            wandb.log({
                "Val_Loss": avg_val_loss,
                "Val Perplexity": perplexity.item(),
                "Total Tokens Processed": token_count,
                "Step": step,
            })
            print(f"\nStep: {step} | Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity.item():.4f} | Tokens: {token_count}")

        if step % save_checkpoint_iter == 0 and step > 0:
            print(f"Saving checkpoint at step {step}")
            _save_snapshot(model, optimizer, None, None, step)
        
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        # Gradient accumulation loop (can be optimized later)
        for micro_step in range(gradient_accumulation_steps):
            try:
                micro_batch = next(train_data_iterator)
            except StopIteration:
                train_data_iterator = iter(train_dataloader)
                micro_batch = next(train_data_iterator)

            # batch data extraction
            idx = micro_batch['input_ids'].to(device)
            targets_idx = micro_batch['decoder_input_ids'].to(device)
            targets = micro_batch['labels'].to(device)
            
            src_mask = torch.ones(batch_size, block_size, dtype=idx.dtype).to(device)
            tgt_mask = torch.ones(batch_size, block_size, dtype=idx.dtype).to(device)
            
            src_mask = src_mask.masked_fill(idx == tokenizer.pad_token_id, 0)
            tgt_mask = tgt_mask.masked_fill(targets_idx == tokenizer.pad_token_id, 0)
            
            token_count += idx.numel()
            
            # Forward pass
            with torch.autocast(device_type=device, dtype=torch.float16):
                loss = model(idx, targets_idx, targets, src_mask, tgt_mask)

            # Scale loss
            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()

            # Backward pass
            scaler.scale(loss).backward()

        lr = get_lr(step)
        for params in optimizer.param_groups:
            params['lr'] = lr
        
        total_norm_before = 0.0
        if clip != 0.0:
            scaler.unscale_(optimizer)
            total_norm_before = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        scaler.step(optimizer)
        scaler.update()
        
        torch.cuda.empty_cache()

        perplexity = torch.exp(torch.tensor(accumulated_loss))

        wandb.log({
            "Learning Rate": lr,
            "Train_Loss": accumulated_loss,
            "Train Perplexity": perplexity.item(),
            "Total Tokens Processed": token_count,
            "Step": step,
            "Gradient Norm": total_norm_before.item() if hasattr(total_norm_before, 'item') else total_norm_before,
        })

        if step > 0 and step % 500 == 0:
            model.eval()
            prompts = ["Hello! Myself an AI Assistant. How are you?", 
                      "My name is Khan", 
                      "How are you?", 
                      "The AI will take our jobs ahhh!"]
            
            with open(f'/content/generations/generations_{step}.txt', 'w') as f:
                f.write(f"{'='*60}\n")
                f.write(f"Step: {step}\n")
                f.write(f"{'='*60}\n\n")
                
                for prt in prompts:
                    print(f"\ntext for prompt: {prt}")
                    try:
                        generated_text = topk_sampling(model, prt, str(device), max_length=block_size//2, 
                                                      top_k=100, temperature=0.9, repetition_penalty=1.2)
                        beam_text = beam_search_corrected(model, prt, tokenizer, str(device), 
                                                         block_size=block_size, beam_width=5, 
                                                         max_length=block_size//2, temperature=0.8)

                        print(f"Top-K Generated: {generated_text}")
                        print(f"Beam Search: {beam_text}")
                        
                        f.write(f"Prompt: {prt}\n")
                        f.write(f"Top-K Generated: {generated_text}\n")
                        f.write(f"Beam Search: {beam_text}\n")
                        f.write(f"{'-'*60}\n\n")
                    except Exception as e:
                        print(f"Error generating text: {e}")
                        f.write(f"Error for prompt '{prt}': {e}\n\n")
            
            model.train()

    _save_snapshot(model, optimizer, None, None, step)
    wandb.finish()
    print("Training done")

#main execute code

if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"Vocab Size: {src_vocab_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Block Size: {block_size}")
    print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    
    train()
