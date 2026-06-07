from pathlib import Path
import torch
import os
import importlib
import time


def retry(fn, attempts=5):
    def wrapper(*args, **kwargs):
        delay = 1
        for attempt in range(attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt < attempts - 1:
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError(f"Failed to execute {fn.__name__} after {attempts} attempts")
    return wrapper


@retry
def load_cfg(cfg_path: str) -> dict:
    mod = importlib.import_module(cfg_path)
    cfg = getattr(mod, "CFG", None) # имя - строго CFG
    assert cfg is not None and isinstance(cfg, dict)
    return cfg


@retry
def _save(obj, path, accelerator=None):
    path = Path(path)
    if accelerator is None or accelerator.is_main_process:
        path.parent.mkdir(parents=True, exist_ok=True)
    if accelerator is None:
        torch.save(obj, path)
    else:
        accelerator.save(obj, path)


@retry
def _model_state_dict(model, accelerator=None):
    if accelerator is None:
        return model.state_dict()
    return accelerator.get_state_dict(model)


def PSNR(x, y, data_range=2.0, eps=1e-8):
    mse = torch.mean((x - y) ** 2)
    mse = torch.clamp(mse, min=eps)
    return 10 * torch.log10((data_range ** 2) / mse)
