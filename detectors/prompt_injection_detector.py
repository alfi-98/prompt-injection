from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

@dataclass
class PIDetectorConfig:
    model_id: str = "protectai/deberta-v3-base-prompt-injection-v2"
    max_length: int = 512
    threshold: float = 0.5
    device: int = -1
    cache_dir: str | None = None

class PromptInjectionDetector:
    def __init__(self, cfg: PIDetectorConfig = PIDetectorConfig()):
        self.cfg = cfg
        tok = AutoTokenizer.from_pretrained(cfg.model_id, cache_dir=cfg.cache_dir)
        mdl = AutoModelForSequenceClassification.from_pretrained(cfg.model_id, cache_dir=cfg.cache_dir)
        self.pipe = TextClassificationPipeline(
            model=mdl, tokenizer=tok, device=cfg.device, top_k=None, function_to_apply="sigmoid"
        )

    def score(self, text: Union[str, List[str]]) -> List[Dict[str, Any]]:
        texts = [text] if isinstance(text, str) else text
        outs = self.pipe(texts, truncation=True, max_length=self.cfg.max_length)
        res = []
        for out in outs:
            scores = {d["label"]: float(d["score"]) for d in out}
            p = scores.get("1", 0.0)
            res.append({"prob_injection": p, "label": int(p >= self.cfg.threshold)})
        return res
