# **Transformer Translation Model

## **Overview**

This project implements a **from-scratch Transformer encoder–decoder translation model** trained on the **ai4bharat/samanantar** parallel corpus (Hindi–English).
It uses:

* Custom-built encoder, decoder, attention heads, embeddings, and positional encodings
* Liger fused cross-entropy loss (optional)
* PyTorch AMP + gradient accumulation + cosine LR decay
* Hugging Face datasets + tokenizers
* Weights & Biases for logging
* Checkpointing, beam search, top-k sampling

The training loop fully supports **long sequence training**, **GPU acceleration**, and **large-batch gradient accumulation**.

---

## **Features**

* Custom Transformer architecture (encoder + decoder)
* Masked self-attention, full attention, cross-attention
* Liger fused Linear + CE loss for faster training
* Mixed precision (AMP)
* Cosine LR scheduler with warmup
* Gradient accumulation for large effective batch sizes
* Beam search + top-k sampling inference
* Hugging Face dataset integration
* W&B experiment tracking
* Automatic checkpoint saving
* Generation logs output every 500 steps

---

## **Dataset**

This project uses:

**Dataset:** `ai4bharat/samanantar`
**Language pair:** Hindi (hi)
**Split:** 99% train / 1% validation

```
load_dataset("ai4bharat/samanantar", "hi")
```

**Tokenizer:**
`ai4bharat/IndicBARTSS` (AlbertTokenizer-based)

---

## **Environment Setup**

### **Install Dependencies**

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets wandb tqdm torchinfo
pip install liger-kernel
pip install accelerate
```

### **Login**

```
wandb login
huggingface-cli login
```

Provide your HF_TOKEN as an environment variable if dataset requires auth.

---

## **Training**

Run:

```
python main.py
```

The script:

* Loads the dataset
* Builds encoder + decoder
* Compiles the model (torch.compile) if available
* Logs to W&B
* Saves checkpoints under: `/content/checkpoints`
* Saves generated outputs under: `/content/generations`

**Hyperparameters:**
Defined in the config block (batch size, LR, warmup, etc.).

**Important values:**

* `block_size = 512`
* `batch_size = 32`
* `no_of_decoder_layers = 6`
* `embeddings_dims = 512`
* `use_liger = True` (set False if not using Liger kernels)

---

## **Model Architecture**

A custom implementation of:

* **Encoder**

  * Token embeddings
  * Positional embeddings
  * Full multi-head self-attention
  * MLP block + LayerNorm

* **Decoder**

  * Masked self-attention
  * Encoder–decoder cross-attention
  * MLP block + LayerNorm
  * Positional embeddings

* **Output layer**

  * Linear projection → vocabulary

* **Loss**

  * LigerFusedLinearCrossEntropyLoss (optional)
  * CrossEntropy fallback

---

## **Inference**

Two generation methods are supported:

### **Top-k Sampling**

```
topk_sampling(model, prompt, device, max_length=30, top_k=50)
```

### **Beam Search**

```
beam_search_corrected(model, prompt, tokenizer, device, block_size)
```

---

## **Checkpoints**

Checkpoints are saved every **500 steps**:

```
/content/checkpoints/snapshot_{step}.pt
```

They contain:

* Model state dict
* Optimizer state
* Step

---

## **Logging**

Everything is logged to **Weights & Biases**, including:

* Learning rate
* Train loss
* Validation loss
* Perplexity
* Gradient norms
* Token count
* Generation samples

---

## **Project Structure**

Recommended folder structure when committing:

```
project/
│
├── main.py
├── README.md
├── requirements.txt
├── checkpoints/          # auto-created
├── generations/          # auto-created
└── ...
```

---

## **requirements.txt (optional)**

```
torch
torchvision
torchaudio
transformers
datasets
wandb
tqdm
liger-kernel
accelerate
torchinfo
```

---

## **Notes**

* Set your `HF_TOKEN` in environment variables (do not commit it).
* GPU strongly recommended.
* To disable Liger kernels, set `use_liger = False`.

---

If you want, I can also produce:

* a shorter README
* a more polished + professional open-source style README
* a version with badges (W&B, Python version, license)
* or convert this into Markdown with formatting ready to paste into GitHub.
