#%%
import ast
import os
import numpy as np
import pandas as pd
import torch
import random
from typing import Any, Dict, List, Optional
#%%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
#%%
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
#%%
def is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    if isinstance(x, str) and x.strip().lower() in {"", "nan", "none", "null", "na"}:
        return True
    return False
#%%
def clean_value(x: Any, default: Optional[str] = None) -> Optional[str]:
    if is_missing(x):
        return default
    return str(x).strip()
#%%
def parse_context(context_value: Any) -> Dict[str, Any]:
    if isinstance(context_value, dict):
        return context_value
    if is_missing(context_value):
        return {}
    try:
        parsed = ast.literal_eval(str(context_value))
        if isinstance(parsed, dict):
            return parsed
        return {}
    except Exception:
        return {}
#%%
def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)