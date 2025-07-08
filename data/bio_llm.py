# %%
from clearml import Task, Logger
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import scanpy as sc
from mammal.model import Mammal
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# %%
# ========== Configuration ==========
use_lora = True

num_epochs = 2
batch_size = 32

learning_rate = 2e-5

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

num_bio_tokens = 1
input_text = "What cell type is this: [BIO] [TRAINABLE_BIO]?"


# %%

# ========== Device Configuration ==========
#device = "mps"  # for Mac
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= ClearML Init ==========
task = Task.init(
    project_name="MAMMAL-Granite",
    task_name="[Interactive] Zero-Shot_Cell_Type_Annotation_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    task_type="training" # Task.TaskTypes.TRAINING
)
logger = Logger.current_logger()
task.connect({
    "use_lora": use_lora,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate
})

# %%
print(device)

# %%
use_subset_ann = False

# ========= Load scRNA-seq data from file ==========
local_root_data_path = '/Users/chinghuei/Downloads'
remote_root_data_path = '/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/'

# human PBMC cell type classification task from scEval
# same dataset is also used for the batch effect task
adata_all = sc.read_h5ad(remote_root_data_path + '/batch_effect/human_pbmc/h5ad/standardized.h5ad')

if use_subset_ann:
    adata = adata_all[adata_all.obs.sample(frac=0.2, random_state=42).index, :]
else:
    adata = adata_all


# %%

# ======== Load MAMMAL Encoder (frozen) ========
mammal_model = Mammal.from_pretrained("ibm/biomed.omics.bl.sm.ma-ted-458m").eval().to(device)
mammal_tokenizer = ModularTokenizerOp.from_pretrained("ibm/biomed.omics.bl.sm.ma-ted-458m")
for p in mammal_model.parameters():
    p.requires_grad = False # Freeze all weights


##/dccstor/bmfm-targets/models/omics/transcriptome/scRNA/pretrain/bmfm.omics.bert.110m.scRNA.multitask.v3

# ======== Load Granite (or LLaMA) Decoder ========
llm_model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.3-2b-base").to(device)
llm_tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-base")

# ========= LoRA ==========
if use_lora:
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    llm_model = get_peft_model(llm_model, lora_config)
else:
    # freeze LLM when not using LoRA (this way only the projection layer is updated)
    # do Not freeze LLM when using LoRA, as q_proj & v_proj are still injected into LLM
    for p in llm_model.parameters():
        p.requires_grad = False


# %%
# --- Inject these before tokenization ---
special_tokens = {'additional_special_tokens': ['[BIO]', '[TRAINABLE_BIO]']}
llm_tokenizer.add_special_tokens(special_tokens)
llm_model.resize_token_embeddings(len(llm_tokenizer))

# --- Define trainable [TRAINABLE_BIO] token embedding ---
trainable_bio_token_embed = nn.Parameter(torch.randn(llm_model.config.hidden_size)).to(device)

# %%
class TrainableBIO(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(dim))

trainable_bio_module = TrainableBIO(llm_model.config.hidden_size).to(device)

# %%

# ======== Adapter: Projects MAMMAL embeddings to LLaMA input space (Trainable projection layer) ========
class MammalToLlamaAdapter(nn.Module):
    """
    Projects MAMMAL encoder outputs to N token embeddings for LLM input.
    If N=1, this is equivalent to projecting a single [CLS] token.
    If N>1, assume input shape is (B, N, input_dim), e.g., top-N gene tokens when using sorted genes.

    input_dim = dimension of bio embeddings (e.g. MAMMAL is 768)
    target_dim = dimension of LLM embeddings (e.g. Granite is 2048)
    """
    def __init__(self, input_dim=768, hidden_dim=1024, target_dim=2048, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens

        # one layer simple projection (as in LLaVA 1)
        # self.proj = nn.Sequential(
        #     nn.Linear(input_dim, target_dim),
        # )

        # two layerer simple MLP (as in LLaVA 2)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )

    def forward(self, x):
        # Input x shape: (B, input_dim) or (B, N, input_dim)
        if x.ndim == 2:  # (B, input_dim) -> (B, 1, input_dim)
            x = x.unsqueeze(1)
            
        return self.proj(x)  # Output shape: (B, N, target_dim)

adapter = MammalToLlamaAdapter(num_tokens = num_bio_tokens).to(device)


# %%
from mammal.keys import (
    ENCODER_INPUTS_STR,
    ENCODER_INPUTS_TOKENS,
    ENCODER_INPUTS_ATTENTION_MASK,
)
# -------------------------------
# PBMC Dataset for AnnData Input
# -------------------------------
class AnnDataset(Dataset):
    def __init__(self, adata, label_key="CellType", num_tokens=4):
        self.adata = adata
        self.labels = adata.obs[label_key].values
        self.genes = adata.var_names.tolist()
        self.num_tokens = num_tokens

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        expr = self.adata.X[idx].toarray().flatten() if hasattr(self.adata.X[idx], 'toarray') else self.adata.X[idx]
        top_n = 1024    # Top expressed genes
        sorted_genes = [gene for _, gene in sorted(zip(expr, self.genes), reverse=True) if _ > 0]

        # MAMMAL expects something like "[BRCA1][TP53][EGFR]"
        top_genes_str = "[" + "][".join(sorted_genes[:top_n]) + "]"

        # Build MAMMAL input format
        sample_dict = dict()
        sample_dict[ENCODER_INPUTS_STR] = f"<@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>{top_genes_str}<EXPRESSION_DATA_END><EOS>"

        # Tokenize for MAMMAL
        mammal_tokenizer(sample_dict=sample_dict,
                         key_in=ENCODER_INPUTS_STR,
                         key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
                         key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK)
        tokens = torch.tensor(sample_dict[ENCODER_INPUTS_TOKENS]).unsqueeze(0).to(device)
        attention = torch.tensor(sample_dict[ENCODER_INPUTS_ATTENTION_MASK]).unsqueeze(0).to(device)
        
        batch_dict = {
            ENCODER_INPUTS_TOKENS: tokens,
            ENCODER_INPUTS_ATTENTION_MASK: attention,
            "forward_mode": "encoder"
        }

        with torch.no_grad():
            output = mammal_model(batch_dict)
            last_hidden = output["model.out.encoder_last_hidden_state"].squeeze(0)
            
            # Get first N (,i.e., num_tokens) token embeddings
            # if N = 1 we got the first token, which is [CLS]

            # --- what we used to do --- 
            # embedding = output["model.out.encoder_last_hidden_state"].mean(dim=1).squeeze(0)
            # --- is not ideal as it's the average of all tokens in the last hidden layerm which also includes [CLS] 
            
            if last_hidden.size(0) < self.num_tokens:
                padding = self.num_tokens - last_hidden.size(0)
                pad = torch.zeros(padding, last_hidden.size(1)).to(device)
                bio_embeddings = torch.cat([last_hidden, pad], dim=0)[:self.num_tokens]
            else:
                bio_embeddings = last_hidden[:self.num_tokens]

        # bio_embeddings is calculated from self.adata.X[idx]        
        return bio_embeddings, self.labels[idx]


# %%
def inject_bio_and_trainable_tokens(
    input_ids, input_embeds, bio_embeddings, adapter, trainable_bio_module, tokenizer
):
    """
    Replace [BIO] and [TRAINABLE_BIO] token positions in input_embeds with:
    - projected bio embeddings (N tokens) at [BIO] positions
    - trainable embedding at [TRAINABLE_BIO] position

    input_ids: (B, T)
    input_embeds: (B, T, D)
    bio_embeddings: (B, N, D_input) --> D_input is the original dimension before projection
    adapter: projection model to map to LLM space
    trainable_bio_token: (1, D)
    """
    bio_token_id = tokenizer.convert_tokens_to_ids("[BIO]")
    trainable_token_id = tokenizer.convert_tokens_to_ids("[TRAINABLE_BIO]")

    B, T, D = input_embeds.shape # input includes text tokens and "[BIO]" and "[TRAINABLE_BIO]", total length = T (tokens)
    N = bio_embeddings.shape[1] # bio embeddings has shape (B, N, D_input)

    projected_bio = adapter(bio_embeddings)  # (B, N, D), the adapter project D_input to D_target (D_target = D)

    for i in range(B):
        # Replace [BIO] tokens
        bio_pos = (input_ids[i] == bio_token_id).nonzero(as_tuple=True)[0]
        for j, pos in enumerate(bio_pos[:N]):  # Up to N tokens
            input_embeds[i, pos] = projected_bio[i, j]

        # Replace [TRAINABLE_BIO] tokens
        trainable_pos = (input_ids[i] == trainable_token_id).nonzero(as_tuple=True)[0]
        for pos in trainable_pos:
            input_embeds[i, pos] = trainable_bio_module.embedding  # shape (D,)
    
    return input_embeds

# %%
import torch

def get_token_embeddings(llm_model, input_ids):
    return llm_model.get_input_embeddings()(input_ids)

def inject_and_tokenize(
    llm_model, tokenizer, prompt_text, labels, bio_embeddings, adapter, trainable_bio_module, device, max_length=64
):
    """
    Tokenizes [prompt + label], injects projected BIO tokens, and returns token ids and final embeddings.
    """
    B = bio_embeddings.size(0)
    bio_embeddings = bio_embeddings.to(device)

    if labels is not None:
        full_texts = [f"{prompt_text} {label}" for label in labels]
    else:
        full_texts = [prompt_text] * B
        
    tokenized = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)
    embeds = get_token_embeddings(llm_model, input_ids)

    final_embeds = inject_bio_and_trainable_tokens(
        input_ids, embeds, bio_embeddings, adapter, trainable_bio_module, tokenizer
    )

    return input_ids, final_embeds, attention_mask

def mask_prompt_loss(input_ids, tokenizer, mask_token="[TRAINABLE_BIO]"):
    """
    Returns loss labels by masking everything before the last [TRAINABLE_BIO] token.
    """
    labels = input_ids.clone()
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    for i in range(input_ids.size(0)):
        seq = input_ids[i]
        pos = (seq == mask_id).nonzero(as_tuple=True)[0]
        start = pos[-1].item() + 1 if len(pos) > 0 else 1
        labels[i, :start] = -100

    return labels


# %%
# ======== Training Function ========
def train_model(llm_model, adapter, dataloader, tokenizer, device, epochs=1, use_lora=True, learn_rate=2e-5):
    # for a decoder model, padding side should be "right" in training and "left" when generating
    tokenizer.padding_side = "right"    # i.e., [PAD] are added at the end

    # always train adapter
    adapter.train()

    # only train LLM if use LoRA
    llm_model.train() if use_lora else llm_model.eval()

    # What is trainable?
    # 1. weights in the projection layer
    # 2. embedding of the special token [TRAINABLE_BIO] 
    # 3. [optional] LoRA for LLM
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(trainable_bio_module.parameters()) + (list(llm_model.parameters()) if use_lora else []),
        lr=learn_rate
    )

    for epoch in range(epochs):
        total_loss = 0
        for step, (bio_embeddings, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            input_ids, final_embeds, attention_mask = inject_and_tokenize(
                llm_model, tokenizer, input_text, labels, bio_embeddings, adapter, trainable_bio_module, device
            )
            labels_full = mask_prompt_loss(input_ids, tokenizer)

            outputs = llm_model(inputs_embeds=final_embeds, labels=labels_full)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            logger.report_scalar("Loss", "Train_Iteration", iteration=epoch * len(dataloader) + step, value=loss.item())

        avg_loss = total_loss / len(dataloader)
        logger.report_scalar("Loss", "Train_Epoch", iteration=epoch, value=avg_loss)

        # Save weights
        adapter_path = checkpoint_dir / f"adapter_epoch{epoch+1}.pt"
        torch.save(adapter.state_dict(), adapter_path)
        task.upload_artifact(name=f"adapter_epoch{epoch+1}", artifact_object=str(adapter_path))
        if use_lora:
            llm_model.save_pretrained(checkpoint_dir / f"llm_epoch{epoch+1}")


# %%
def evaluate_model(llm_model, adapter, dataloader, tokenizer, device):
    adapter.eval()
    llm_model.eval()
    preds, truths = [], []

    tokenizer.padding_side = "left"
    
    with torch.no_grad():
        for bio_embeddings, labels in tqdm(dataloader, desc="Evaluating"):
            B = bio_embeddings.size(0)
            input_ids, input_embeds, attention_mask = inject_and_tokenize(
                llm_model, tokenizer, input_text, None, bio_embeddings, adapter, trainable_bio_module, device
            )

            outputs = llm_model.generate(
                attention_mask = attention_mask,
                inputs_embeds = input_embeds,
                max_length = input_embeds.shape[1] + 20,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = tokenizer.eos_token_id,
                do_sample = False
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            preds.extend([p.strip().lower() for p in decoded])
            truths.extend([t.strip().lower() for t in labels])

    acc = accuracy_score(truths, preds)
    logger.report_scalar("Accuracy", "Eval", iteration=0, value=acc)
    print(f"Evaluation Accuracy: {acc:.4f}")

# %%
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.amp import autocast
import numpy as np

def evaluate_model(llm_model, adapter, dataloader, tokenizer, device):
    adapter.eval()
    llm_model.eval()
    preds, truths = [], []

    tokenizer.padding_side = "left"
    
    with torch.no_grad():
        for batch_idx, (bio_embeddings, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
            with autocast(device_type='cuda'):  # Optional: memory saving
                input_ids, input_embeds, attention_mask = inject_and_tokenize(
                    llm_model, tokenizer, input_text, 
                    labels=None,  # <-- no labels during eval
                    bio_embeddings=bio_embeddings,
                    adapter=adapter,
                    trainable_bio_module=trainable_bio_module,
                    device=device
                )

                outputs = llm_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_length=input_embeds.shape[1] + 20,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )

            # decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # preds.extend([p.strip().lower() for p in decoded])
            # truths.extend([t.strip().lower() for t in labels])

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_preds = [p.strip().lower() for p in decoded]
            decoded_truths = [t.strip().lower() for t in labels]

            for i, (pred, true) in enumerate(zip(decoded_preds, decoded_truths)):
                print(f"[{batch_idx * dataloader.batch_size + i + 1}] pred: {pred:<30} | truth: {true}")

            preds.extend(decoded_preds)
            truths.extend(decoded_truths)

    # === Metrics ===
    acc = accuracy_score(truths, preds)
    macro_f1 = f1_score(truths, preds, average='macro', zero_division=0)

    print("\nFinal Evaluation Metrics:")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(truths, preds, zero_division=0))

    logger.report_scalar("Accuracy", "Eval", iteration=0, value=acc)
    logger.report_scalar("Macro_F1", "Eval", iteration=0, value=macro_f1)

    # confusion matrix
    try:
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay.from_predictions(truths, preds, ax=ax, xticks_rotation=45)
        plt.title("Confusion Matrix")
        task.logger.report_matplotlib_figure("ConfusionMatrix", "Eval", iteration=0, figure=fig)
        plt.show()
    except Exception as e:
        print(f"Could not display confusion matrix: {e}")


# %%

def run_inference(llm_model, adapter, tokenizer, prompt_text, mammal_embedding, device):
    llm_model.eval()
    adapter.eval()

    with torch.no_grad():
        bio_embeddings = mammal_embedding.unsqueeze(0).to(device)  # (1, N, D)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        prompt_embeds = get_token_embeddings(llm_model, prompt_ids)

        input_embeds = inject_bio_and_trainable_tokens(
            prompt_ids, prompt_embeds, bio_embeddings, adapter, trainable_bio_module, tokenizer
        )

        # only inputs_embeds is needed (i.e. no inputs_ids) 
        # because we are doing 1 sample at a time (i.e. no batching) and there is no padding, 
        # and no need for attention_mask (all tokens are real and assumes full attention)
        generated_ids = llm_model.generate(
            inputs_embeds=input_embeds,
            max_length=input_embeds.shape[1] + 20,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

        return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

# %%

# ======== RUN EVERYTHING ========
# 1. Create full dataset
dataset = AnnDataset(adata, label_key="CellType", num_tokens=num_bio_tokens)

### dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 2. Define split sizes
total_size = len(dataset)
train_size = int(0.6 * total_size)
dev_size = int(0.2 * total_size)
test_size = total_size - train_size - dev_size  # to catch rounding errors

# 3. Random split
train_dataset, dev_dataset, test_dataset = random_split(
    dataset, 
    [train_size, dev_size, test_size],
    generator=torch.Generator().manual_seed(42)  # reproducible split
)

# 4. Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %%
run_training_mode = True
run_evaluation_mode = True
run_inference_mode = True

# %%
if run_training_mode:
    # for a decoder model, padding side should be "right" in training and "left" when generating
    llm_tokenizer.padding_side = "right"
    train_model(llm_model, adapter, train_loader, llm_tokenizer, device, epochs=num_epochs, use_lora=use_lora, learn_rate=learning_rate)

    # train_model_early_stop(llm_model, adapter, llm_tokenizer, train_loader, dev_loader,
    #     device, 
    #     epochs=num_epochs, 
    #     use_lora=use_lora, 
    #     learn_rate=learning_rate,
    #     early_stopping_patience=3, 
    #     max_grad_norm=1.0,
    #     logger=logger)

    # Final save directory
    final_model_dir = checkpoint_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    # Save adapter (projection layer)
    adapter_path = final_model_dir / "mammal_to_llm_adapter.pt"
    torch.save(adapter.state_dict(), adapter_path)
    #task.upload_artifact(name="final_adapter", artifact_object=str(adapter_path))

    # Resize and make sure config reflects it (added [BIO] and other special tokens)
    llm_model.config.vocab_size = len(llm_tokenizer)  # e.g., 49154 with added special tokens
    llm_model.resize_token_embeddings(len(llm_tokenizer))  # Adjust the embedding layer

    # Save LLM (optionally with LoRA)
    if use_lora:
        llm_model_merged_with_lora = llm_model.merge_and_unload()
        llm_model_merged_with_lora.resize_token_embeddings(len(llm_tokenizer))

        llm_model_merged_with_lora.save_pretrained(final_model_dir / "granite_lora")
    #    task.upload_artifact(name="final_lora_llm", artifact_object=str(final_model_dir / "granite_lora"))
    else:
        # Save model and config for non-LoRA zero-shot inference
        llm_model.save_pretrained(final_model_dir / "granite")
    #    task.upload_artifact(name="final_llm", artifact_object=str(final_model_dir / "granite"))

    # Save tokenizer
    llm_tokenizer.save_pretrained(final_model_dir / "tokenizer")
    #task.upload_artifact(name="llm_tokenizer", artifact_object=str(final_model_dir / "tokenizer"))


# %%
import torch
import gc

# Delete variables
del llm_model_merged_with_lora  # del model

# Collect garbage
gc.collect()

# Release unreferenced memory held by PyTorch
torch.cuda.empty_cache()

# %%
if run_evaluation_mode:
    llm_tokenizer.padding_side = "left"
    evaluate_model(llm_model, adapter, test_loader, llm_tokenizer, device)

    del test_loader
    gc.collect()
    torch.cuda.empty_cache()

# %%
len(llm_tokenizer)

# %%
run_inference_mode = True
final_model_dir = checkpoint_dir / "final_model"

# %%
if run_inference_mode:
    final_model_dir = checkpoint_dir / "final_model"

    # === Load the tokenizer first ===
    llm_tokenizer = AutoTokenizer.from_pretrained(final_model_dir / "tokenizer")
#    llm_tokenizer.add_special_tokens(special_tokens)
    llm_tokenizer.padding_side = "left" # default in inference mode, [PAD]s are added to the beginning 

    # == Load the config.json from saved model ===
    # This is important as we added new special tokens and 
    #    the # of tokens in the new model (e.g. 49,154) <--- saved in the model’s embedding.weight (loaded from checkpoint)
    #    the # of tokens in the base model (e.g. 49,152) <--- model’s config (from config.json): says vocab_size = 49152 (without special tokens)
    # config.json should be saved if we 
    #   1. do not use LoRA
    #   2. use LoRA and merge it back to the base
    # This is because Hugging Face does not create a new config.json for PEFT adapters like LoRA
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(final_model_dir / "granite_lora")

    # === Load final saved model ===
    llm_model = AutoModelForCausalLM.from_pretrained(
        final_model_dir / "granite_lora",
        config=config).to(device)
    llm_model.resize_token_embeddings(len(llm_tokenizer))
    
    adapter = MammalToLlamaAdapter(num_tokens=1).to(device)
    adapter.load_state_dict(torch.load(final_model_dir / "mammal_to_llm_adapter.pt"))
    adapter.eval()

    logger = Logger.current_logger()
    prompt_text = "Given [BIO], what is the most likely cell type [TRAINABLE_BIO]?"

    # === Get 50 random cells from AnnData ===
    import random
    indices = random.sample(range(len(adata)), 50)
    subset = adata[indices]

    for i, idx in enumerate(subset.obs.index):
        # Preprocess cell into MAMMAL input
        expr = subset.X[i].toarray().flatten() if hasattr(subset.X[i], 'toarray') else subset.X[i]
        sorted_genes = [gene for _, gene in sorted(zip(expr, subset.var_names), reverse=True) if _ > 0]
        
        # MAMMAL expects something like "[BRCA1][TP53][EGFR]"
        top_n = 1024
        top_genes_str = "[" + "][".join(sorted_genes[:top_n]) + "]"

        # Build MAMMAL input format
        sample_dict = dict()
        sample_dict[ENCODER_INPUTS_STR] = f"<@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>{top_genes_str}<EXPRESSION_DATA_END><EOS>"

        # Tokenize for MAMMAL
        mammal_tokenizer(sample_dict=sample_dict,
                         key_in=ENCODER_INPUTS_STR,
                         key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
                         key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK)
        tokens = torch.tensor(sample_dict[ENCODER_INPUTS_TOKENS]).unsqueeze(0).to(device)
        attention = torch.tensor(sample_dict[ENCODER_INPUTS_ATTENTION_MASK]).unsqueeze(0).to(device)

        batch_dict = {
            ENCODER_INPUTS_TOKENS: tokens,
            ENCODER_INPUTS_ATTENTION_MASK: attention,
            "forward_mode": "encoder"
        }

        with torch.no_grad():
            output = mammal_model(batch_dict)

            last_hidden = output["model.out.encoder_last_hidden_state"].squeeze(0)
            if last_hidden.size(0) < adapter.num_tokens:
                padding = adapter.num_tokens - last_hidden.size(0)
                pad = torch.zeros(padding, last_hidden.size(1)).to(device)
                bio_embeddings = torch.cat([last_hidden, pad], dim=0)[:adapter.num_tokens]
            else:
                bio_embeddings = last_hidden[:adapter.num_tokens]

        # === Run inference ===
        prediction = run_inference(llm_model, adapter, llm_tokenizer, prompt_text, bio_embeddings, device)

        # === Log to ClearML ===
        logger.report_text(f"[{i+1}] Predicted: {prediction} | Ground Truth: {subset.obs['CellType'][i]}")
