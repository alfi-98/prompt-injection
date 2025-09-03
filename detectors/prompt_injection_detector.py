from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Union

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class PIDetectorConfig:
    model_id: str = "protectai/deberta-v3-base-prompt-injection-v2"
    max_length: int = 512
    threshold: float = 0.5
    device: int = -1
    cache_dir: str | None = None
    batch_size: int = 8

class PromptInjectionDetector:
    def __init__(self, cfg: PIDetectorConfig = PIDetectorConfig()):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, cache_dir=cfg.cache_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_id, cache_dir=cfg.cache_dir)
        self.model.eval()
        self.device = torch.device("cpu" if cfg.device == -1 else f"cuda:{cfg.device}")
        self.model.to(self.device)
        self.num_labels = int(self.model.config.num_labels or 2)

    def _softmax(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1)

    def score(self, text: Union[str, List[str]]) -> List[Dict[str, Any]]:
        texts = [text] if isinstance(text, str) else list(text)
        out: List[Dict[str, Any]] = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = texts[i:i + self.cfg.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.cfg.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = self._softmax(logits).detach().cpu()
            for j in range(probs.size(0)):
                p_inj = float(probs[j, 1] if self.num_labels >= 2 else probs[j, 0])
                out.append({"prob_injection": p_inj, "label": int(p_inj >= self.cfg.threshold)})
        return out
