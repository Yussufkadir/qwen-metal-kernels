import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENGINE_DIR = ROOT / "kernel/qwen_kernel/qwen_kernel"
ENGINE = ENGINE_DIR / "inference_engine"
WEIGHTS_DIR = ENGINE_DIR / "qwen_weights"

def engine_available() -> bool:
    return ENGINE.is_file() and (WEIGHTS_DIR / "embed_tokens.weight.bin").is_file()

def backend_available(backend: str) -> bool:
    if not engine_available():
        return False
    probe = WEIGHTS_DIR / "layer0.mlp.down_proj.weight.bin"
    if backend in ("int8", "int8-fp16"):
        probe = WEIGHTS_DIR / "layer0.mlp.down_proj.weight.q8.bin"
    if backend in ("int4", "mixed", "hybrid"):
        probe = WEIGHTS_DIR / "layer0.mlp.down_proj.weight.q4.bin"
    return probe.is_file()

def load_tokenizer_or_none():
    import qwen_chat
    try:
        return qwen_chat.load_fast_qwen_tokenizer(qwen_chat.MODEL_NAME)
    except Exception:
        return None
