import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from mammal.model import Mammal
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from mammal.keys import (
    ENCODER_INPUTS_STR,
    ENCODER_INPUTS_TOKENS,
    ENCODER_INPUTS_ATTENTION_MASK,
)
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainableBIO(nn.Module):
    """Simple container for a trainable embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(dim))


def get_token_embeddings(llm_model, input_ids):
    return llm_model.get_input_embeddings()(input_ids)


class MammalToLlamaAdapter(nn.Module):
    """Project MAMMAL encoder output to LLM embedding space."""

    def __init__(self, input_dim=768, hidden_dim=1024, target_dim=2048, num_tokens=1):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.proj(x)


def inject_bio_and_trainable_tokens(
    input_ids,
    input_embeds,
    bio_embeddings,
    adapter,
    trainable_bio_module,
    tokenizer,
):
    """Replace [BIO] and [TRAINABLE_BIO] tokens with projected embeddings."""
    bio_token_id = tokenizer.convert_tokens_to_ids("[BIO]")
    trainable_token_id = tokenizer.convert_tokens_to_ids("[TRAINABLE_BIO]")

    projected_bio = adapter(bio_embeddings)

    for i in range(input_ids.size(0)):
        bio_pos = (input_ids[i] == bio_token_id).nonzero(as_tuple=True)[0]
        for j, pos in enumerate(bio_pos[: projected_bio.size(1)]):
            input_embeds[i, pos] = projected_bio[i, j]

        trainable_pos = (input_ids[i] == trainable_token_id).nonzero(as_tuple=True)[0]
        for pos in trainable_pos:
            input_embeds[i, pos] = trainable_bio_module.embedding

    return input_embeds


def run_inference(
    llm_model,
    adapter,
    tokenizer,
    prompt_text,
    mammal_embedding,
    trainable_bio_module,
    device,
):
    """Generate an answer given prompt and MAMMAL embedding."""
    llm_model.eval()
    adapter.eval()
    with torch.no_grad():
        bio_embeddings = mammal_embedding.unsqueeze(0).to(device)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        prompt_embeds = get_token_embeddings(llm_model, prompt_ids)

        input_embeds = inject_bio_and_trainable_tokens(
            prompt_ids,
            prompt_embeds,
            bio_embeddings,
            adapter,
            trainable_bio_module,
            tokenizer,
        )

        generated = llm_model.generate(
            inputs_embeds=input_embeds,
            max_length=input_embeds.shape[1] + 20,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


# ---------------------- Model Loading & Utilities ----------------------

def load_models(model_dir: Path = Path("checkpoints/final_model"), num_tokens: int = 1):
    """Load tokenizer, LLM, adapter and MAMMAL encoder for inference."""
    llm_tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    llm_tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(model_dir / "granite_lora")
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_dir / "granite_lora", config=config
    ).to(device)
    llm_model.resize_token_embeddings(len(llm_tokenizer))

    adapter = MammalToLlamaAdapter(num_tokens=num_tokens).to(device)
    adapter.load_state_dict(
        torch.load(model_dir / "mammal_to_llm_adapter.pt", map_location=device)
    )
    adapter.eval()

    mammal_model = Mammal.from_pretrained("ibm/biomed.omics.bl.sm.ma-ted-458m").eval().to(device)
    mammal_tokenizer = ModularTokenizerOp.from_pretrained("ibm/biomed.omics.bl.sm.ma-ted-458m")

    trainable_bio_module = TrainableBIO(llm_model.config.hidden_size).to(device)

    return {
        "llm_model": llm_model,
        "llm_tokenizer": llm_tokenizer,
        "adapter": adapter,
        "mammal_model": mammal_model,
        "mammal_tokenizer": mammal_tokenizer,
        "trainable_bio_module": trainable_bio_module,
        "device": device,
    }


def cell_embedding(adata_cell, models, top_n: int = 1024):
    """Convert a single AnnData row to a MAMMAL embedding."""
    expr = adata_cell.X.toarray().flatten() if hasattr(adata_cell.X, "toarray") else adata_cell.X
    genes = adata_cell.var_names
    sorted_genes = [g for v, g in sorted(zip(expr, genes), reverse=True) if v > 0]
    top_genes_str = "[" + "][".join(sorted_genes[:top_n]) + "]"
    sample_dict = {
        ENCODER_INPUTS_STR: f"<@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>{top_genes_str}<EXPRESSION_DATA_END><EOS>"
    }
    models["mammal_tokenizer"](
        sample_dict=sample_dict,
        key_in=ENCODER_INPUTS_STR,
        key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
        key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
    )
    tokens = torch.tensor(sample_dict[ENCODER_INPUTS_TOKENS]).unsqueeze(0).to(models["device"])
    attention = torch.tensor(sample_dict[ENCODER_INPUTS_ATTENTION_MASK]).unsqueeze(0).to(models["device"])
    batch_dict = {
        ENCODER_INPUTS_TOKENS: tokens,
        ENCODER_INPUTS_ATTENTION_MASK: attention,
        "forward_mode": "encoder",
    }
    with torch.no_grad():
        output = models["mammal_model"](batch_dict)
        last_hidden = output["model.out.encoder_last_hidden_state"].squeeze(0)
        if last_hidden.size(0) < models["adapter"].num_tokens:
            padding = models["adapter"].num_tokens - last_hidden.size(0)
            pad = torch.zeros(padding, last_hidden.size(1)).to(models["device"])
            bio_embeddings = torch.cat([last_hidden, pad], dim=0)[: models["adapter"].num_tokens]
        else:
            bio_embeddings = last_hidden[: models["adapter"].num_tokens]
    return bio_embeddings


def answer_question(selected, question: str, models):
    """Return predictions for each selected cell given a question."""
    results = []
    prompt = f"Given [BIO], {question} [TRAINABLE_BIO]?"
    for i in range(selected.shape[0]):
        cell = selected[i : i + 1]
        bio_emb = cell_embedding(cell, models)
        pred = run_inference(
            models["llm_model"],
            models["adapter"],
            models["llm_tokenizer"],
            prompt,
            bio_emb,
            models["trainable_bio_module"],
            models["device"],
        )
        results.append(pred)
    return results
