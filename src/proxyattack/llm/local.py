"""Local LLM backend (Hugging Face transformers on the local GPU), with OpenRouter as the fallback.

Used by the retraining-world preparation (no paid calls when a local model is available):
  * agent harness: the next-token distribution over listing numbers after "ANSWER: " gives the
    agent's exact choice probabilities on a slate in one forward pass (temperature 1);
  * item embeddings for the LoRec calibrator (mean-pooled hidden states);
  * sentence embeddings of titles for the text-aware ranker (UniSRec stand-in).

Env: LLM_BACKEND = auto | local | openrouter (auto: local when CUDA and the model files exist),
LOCAL_LLM_MODEL (default Qwen/Qwen2.5-7B-Instruct), LOCAL_TEXT_ENCODER (default
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). Models are read from the local
Hugging Face cache only (no download).
"""
from __future__ import annotations

import os

import numpy as np

from .openrouter import ChatClient

DEFAULT_LOCAL_LLM = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TEXT_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def local_available(model=None):
    try:
        import torch
        from huggingface_hub import snapshot_download
        if not torch.cuda.is_available():
            return False
        snapshot_download(model or os.environ.get("LOCAL_LLM_MODEL", DEFAULT_LOCAL_LLM), local_files_only=True)
        return True
    except Exception:
        return False


def backend():
    b = os.environ.get("LLM_BACKEND", "auto").strip().lower()
    if b == "auto":
        return "local" if local_available() else "openrouter"
    if b not in ("local", "openrouter"):
        raise ValueError(f"LLM_BACKEND must be auto, local or openrouter, got {b}")
    return b


class LocalLM(ChatClient):
    """A local causal LM: chat generation (ChatClient interface), choice distributions, embeddings."""

    def __init__(self, model=None, device="cuda", dtype="bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = model or os.environ.get("LOCAL_LLM_MODEL", DEFAULT_LOCAL_LLM)
        self.tok = AutoTokenizer.from_pretrained(self.name, local_files_only=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.name, torch_dtype=getattr(torch, dtype),
                                                          local_files_only=True).to(device).eval()
        self.device = device
        self.calls = 0

    def _prompt(self, messages, prefix=""):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix

    def chat(self, messages, model=None, temperature=1.0, replicate=0, max_tokens=None) -> str:
        import torch
        enc = self.tok([self._prompt(messages)], return_tensors="pt").to(self.device)
        torch.manual_seed(replicate)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_tokens or 64, do_sample=temperature > 0,
                                      temperature=max(temperature, 1e-5), pad_token_id=self.tok.pad_token_id)
        self.calls += 1
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def choice_distribution(self, messages_list, k, prefix="ANSWER: ", batch_size=8):
        """(len, k) probabilities of answering 1..k after `prefix` (renormalized over the k numbers)
        and (len,) the probability mass those numbers carry before renormalization."""
        import torch
        ids = [self.tok.encode(str(i + 1), add_special_tokens=False) for i in range(k)]
        if any(len(t) != 1 for t in ids):
            raise ValueError("listing numbers must be single tokens")
        ids = torch.tensor([t[0] for t in ids], device=self.device)
        probs, mass = [], []
        for st in range(0, len(messages_list), batch_size):
            texts = [self._prompt(m, prefix) for m in messages_list[st:st + batch_size]]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1].float()
            p = torch.softmax(logits, -1)[:, ids]
            mass.append(p.sum(1).cpu().numpy())
            probs.append((p / p.sum(1, keepdim=True)).cpu().numpy())
            self.calls += len(texts)
        return np.concatenate(probs), np.concatenate(mass)

    def embed(self, texts, batch_size=16, max_length=256):
        """Mean-pooled final hidden states (one vector per text)."""
        import torch
        out = []
        for st in range(0, len(texts), batch_size):
            enc = self.tok(texts[st:st + batch_size], return_tensors="pt", padding=True, truncation=True,
                           max_length=max_length).to(self.device)
            with torch.no_grad():
                h = self.model(**enc, output_hidden_states=True).hidden_states[-1].float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
        return np.concatenate(out)


def encode_sentences(texts, model=None, device=None, batch_size=64):
    """Sentence-transformers-style mean-pooled embeddings with a local encoder (no download)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = model or os.environ.get("LOCAL_TEXT_ENCODER", DEFAULT_TEXT_ENCODER)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    enc_model = AutoModel.from_pretrained(name, local_files_only=True).to(device).eval()
    out = []
    for st in range(0, len(texts), batch_size):
        enc = tok(texts[st:st + batch_size], return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to(device)
        with torch.no_grad():
            h = enc_model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out)


def sampled_choice_distribution(client: ChatClient, model, messages_list, k, n=8, parse=None):
    """OpenRouter fallback for `choice_distribution`: n sampled answers per slate (costs n calls each)."""
    from .harness import parse_choice
    parse = parse or parse_choice
    probs = np.zeros((len(messages_list), k))
    for i, msgs in enumerate(messages_list):
        for rep in range(n):
            c = parse(client.chat(msgs, model=model, temperature=1.0, replicate=rep), k)
            if c is not None and c >= 0:
                probs[i, c] += 1
    tot = probs.sum(1, keepdims=True)
    return np.where(tot > 0, probs / np.maximum(tot, 1), 1.0 / k), (tot[:, 0] / n)
