# ComfyUI-QwenVL (GGUF)
# GGUF nodes powered by llama.cpp for Qwen-VL models, including Qwen3-VL and Qwen2.5-VL.
# Provides vision-capable GGUF inference and prompt execution.
#
# Models are loaded via llama-cpp-python and configured through gguf_models.json.
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-QwenVL

import gc
import hashlib
import inspect
import json
import math
import os
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

import folder_paths
import comfy.model_management as model_management
from AILab_OutputCleaner import OutputCleanConfig, clean_model_output
from AILab_QwenVL_PromptKit import (
    FL2VA_MIN_CTX,
    FL2VA_MIN_MAX_TOKENS,
    FL2VA_PRESET,
    REF2VA_CTX_PER_IMAGE,
    REF2VA_MAX_CTX,
    REF2VA_MAX_IMAGES,
    REF2VA_IMAGE_MAX_SIDE,
    REF2VA_MIN_CTX,
    REF2VA_MIN_MAX_TOKENS,
    REF2VA_PRESET,
    build_fl2va_request as _build_fl2va_request,
    build_ref2va_request as _build_ref2va_request,
    get_duration_input as _get_duration_input,
    iter_batch_frames as _iter_batch_frames,
    load_fl2va_system_prompt as _load_fl2va_system_prompt,
    load_ref2va_system_prompt as _load_ref2va_system_prompt,
    resolve_preset_text as _resolve_preset_text,
    select_fl2va_frames as _select_fl2va_frames,
    unified_preset_prompts,
)
from AILab_Utils import (
    PLUGIN_DIR,
    GGUF_CONFIG_PATH,
    CUSTOM_MODELS_PATH,
    normalize_input_path as _normalize_input_path,
    parse_model_path_payload as _parse_model_path_payload,
    safe_dirname,
    resolve_base_dir,
    find_local_gguf_file,
    get_comfyui_llm_paths,
    model_name_to_filename_candidates,
    filter_kwargs_for_callable,
    parse_gguf_repos,
    tensor_to_base64_png,
    sample_video_frames,
    resolve_safe_video_max_side,
)
from comfy.utils import ProgressBar



def _sanitize_control_values(device, gpu_layers, duration):
    """Repair widget values shifted by stale node layouts in saved workflows."""
    shifted_duration = None
    if not isinstance(device, str):
        if isinstance(device, (int, float)) and not isinstance(device, bool):
            shifted_gpu_layers = int(device)
            if isinstance(gpu_layers, (int, float)) and not isinstance(gpu_layers, bool):
                candidate = float(gpu_layers)
                if math.isfinite(candidate) and candidate > 0:
                    shifted_duration = candidate
            gpu_layers = shifted_gpu_layers
            print(
                f"[QwenVL] Warning: invalid 'device' widget value ({device}) detected; "
                "treating it as gpu_layers and using device=auto. "
                "Refresh or recreate the node to realign its widgets."
            )
        else:
            print(f"[QwenVL] Warning: invalid 'device' value {device!r}; using auto.")
        device = "auto"
    if not device:
        device = "auto"
    if shifted_duration is not None:
        duration = shifted_duration
    return device, gpu_layers, duration


@dataclass(frozen=True)
class GGUFVLResolved:
    display_name: str
    repo_id: str | None
    alt_repo_ids: list[str]
    author: str | None
    repo_dirname: str
    model_filename: str
    mmproj_filename: str | None
    context_length: int
    image_max_tokens: int
    n_batch: int
    gpu_layers: int
    top_k: int
    pool_size: int


def _load_gguf_vl_catalog():
    flattened: dict[str, dict] = {}
    seen_display_names: set[str] = set()
    base_dir = "LLM/GGUF"

    if GGUF_CONFIG_PATH.exists():
        try:
            with open(GGUF_CONFIG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            base_dir = data.get("base_dir") or base_dir
            for key in ["qwenVL_model", "Qwen_model"]:
                repos = data.get(key) or {}
                if isinstance(repos, dict):
                    parse_gguf_repos(repos, flattened, seen_display_names)
            legacy_models = data.get("models") or {}
            if isinstance(legacy_models, dict):
                for name, entry in legacy_models.items():
                    if isinstance(entry, dict):
                        flattened[name] = entry
        except Exception as exc:
            print(f"[QwenVL] gguf_models.json load failed: {exc}")

    # Merge custom_models.json (central custom config)
    if CUSTOM_MODELS_PATH.exists():
        try:
            with open(CUSTOM_MODELS_PATH, "r", encoding="utf-8") as fh:
                custom_data = json.load(fh) or {}
            for key in ["gguf_models", "gguf_vl_models", "gguf_text_models"]:
                repos = custom_data.get(key) or {}
                if isinstance(repos, dict) and repos:
                    parse_gguf_repos(repos, flattened, seen_display_names, overwrite_existing=True)
        except Exception as exc:
            print(f"[QwenVL] custom_models.json (GGUF) skipped: {exc}")

    return {"base_dir": base_dir, "models": flattened}


def reload_gguf_vl_catalog():
    global GGUF_VL_CATALOG
    GGUF_VL_CATALOG = _load_gguf_vl_catalog()


GGUF_VL_CATALOG = _load_gguf_vl_catalog()


def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return dict(kwargs)

    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(kwargs)

    allowed: set[str] = set()
    for p in params:
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            allowed.add(p.name)
    return {k: v for k, v in kwargs.items() if k in allowed}


def _handler_supports_param(handler_cls, name: str) -> bool:
    """Check whether a chat handler accepts a named parameter (including base classes)."""
    try:
        for klass in getattr(handler_cls, "__mro__", (handler_cls,)):
            if name in inspect.signature(klass.__init__).parameters:
                return True
    except Exception:
        return False
    return False


# Aliases for shared utilities from AILab_Utils
_tensor_to_base64_png = tensor_to_base64_png
_resolve_safe_video_max_side = resolve_safe_video_max_side
_sample_video_frames = sample_video_frames


def _llama_supports_gpu_offload() -> bool | None:
    """Return whether llama.cpp can offload to a GPU.

    In llama-cpp-python >= 0.4 the CUDA/Metal backends are loaded lazily inside
    Llama(), so this only reports True after a model has been constructed.
    None means the check could not be performed (llama_cpp not importable yet).
    """
    try:
        from llama_cpp import llama_cpp as _lc

        return bool(_lc.llama_supports_gpu_offload())
    except Exception:
        return None


def _pick_device(device_choice: str) -> str:
    if device_choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_choice.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    if device_choice == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _download_single_file(repo_ids: list[str], filename: str, target_path: Path):
    if target_path.exists():
        print(f"[QwenVL] Using cached file: {target_path}")
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None
    for repo_id in repo_ids:
        print(f"[QwenVL] Downloading {filename} from {repo_id} -> {target_path}")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_dir=str(target_path.parent),
                local_dir_use_symlinks=False,
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.exists() and downloaded_path.resolve() != target_path.resolve():
                downloaded_path.replace(target_path)
            if target_path.exists():
                print(f"[QwenVL] Download complete: {target_path}")
            break
        except Exception as exc:
            last_exc = exc
            print(f"[QwenVL] hf_hub_download failed from {repo_id}: {exc}")
    else:
        raise FileNotFoundError(f"[QwenVL] Download failed for {filename}: {last_exc}")

    if not target_path.exists():
        raise FileNotFoundError(f"[QwenVL] File not found after download: {target_path}")


def _build_resolved(
    display_name: str,
    entry: dict,
    model_filename: str,
    mmproj_filename: str | None,
) -> GGUFVLResolved:
    repo_id = entry.get("repo_id")
    alt_repo_ids = entry.get("alt_repo_ids") or []

    author = entry.get("author") or entry.get("publisher")
    repo_dirname = entry.get("repo_dirname") or (
        repo_id.split("/")[-1] if isinstance(repo_id, str) and "/" in repo_id else display_name
    )

    def _int(name: str, default: int) -> int:
        value = entry.get(name, default)
        try:
            return int(value)
        except Exception:
            return default

    return GGUFVLResolved(
        display_name=display_name,
        repo_id=repo_id,
        alt_repo_ids=[str(x) for x in alt_repo_ids if x],
        author=str(author) if author else None,
        repo_dirname=safe_dirname(str(repo_dirname)),
        model_filename=str(model_filename),
        mmproj_filename=str(mmproj_filename) if mmproj_filename else None,
        context_length=_int("context_length", 8192),
        image_max_tokens=_int("image_max_tokens", 4096),
        n_batch=_int("n_batch", 512),
        gpu_layers=_int("gpu_layers", -1),
        top_k=_int("top_k", 0),
        pool_size=_int("pool_size", 4194304),
    )


def _resolve_model_entry(model_name: str) -> GGUFVLResolved:
    catalog = _load_gguf_vl_catalog()
    all_models = catalog.get("models") or {}
    entry = all_models.get(model_name) or {}
    if not entry:
        wanted = model_name_to_filename_candidates(model_name)
        for candidate in all_models.values():
            filename = candidate.get("filename")
            if filename and Path(filename).name in wanted:
                entry = candidate
                break

    model_filename = entry.get("filename")
    mmproj_filename = entry.get("mmproj_filename")

    if not model_filename:
        raise ValueError(f"[QwenVL] gguf_models.json entry missing 'filename' for: {model_name}")

    return _build_resolved(model_name, entry, model_filename, mmproj_filename)


def _resolve_model_entry_for_path(model_path: Path) -> GGUFVLResolved:
    """Build a resolved entry for an arbitrary GGUF file on disk.

    Catalog metadata (context length, batch size, etc.) is reused when the
    filename matches a known entry; otherwise sensible defaults are used.
    The mmproj is always auto-detected next to the model file.
    """
    catalog = _load_gguf_vl_catalog()
    all_models = catalog.get("models") or {}
    entry: dict = {}
    target_name = model_path.name.lower()
    for candidate in all_models.values():
        filename = candidate.get("filename")
        if filename and Path(filename).name.lower() == target_name:
            entry = candidate
            break

    return _build_resolved(
        display_name=model_path.stem,
        entry=entry,
        model_filename=str(model_path),
        mmproj_filename=None,
    )


def _find_mmproj_alongside(model_path: Path) -> Path | None:
    """Find the visual projector that lives next to the model file."""
    folder = model_path.parent
    if not folder.is_dir():
        return None

    try:
        candidates = [p for p in folder.glob("*.gguf") if p.is_file() and "mmproj" in p.name.lower()]
    except Exception:
        return None
    if not candidates:
        return None

    # Exact preferred name (e.g. Qwen3-VL "mmproj-model-bf16.gguf").
    for candidate in candidates:
        if candidate.name.lower() == "mmproj-model-bf16.gguf":
            return candidate

    # Match the projector precision to the model quant when possible.
    model_stem = model_path.stem.lower()
    preferred_tokens = ("q8",) if "q8" in model_stem else ("bf16", "f16")
    for token in preferred_tokens:
        for candidate in candidates:
            if token in candidate.name.lower():
                return candidate

    return sorted(candidates, key=lambda p: p.name.lower())[0]


def _list_text_encoder_gguf_files() -> list[str]:
    """List non-mmproj GGUF files from models/LLM, models/text_encoders and models/clip."""
    names: set[str] = set()
    seen_roots: set[str] = set()
    for folder_name in ("LLM", "text_encoders", "clip_gguf"):
        try:
            roots = folder_paths.get_folder_paths(folder_name)
        except Exception:
            continue
        for root in roots:
            root_key = os.path.normcase(os.path.abspath(root))
            if root_key in seen_roots:
                continue
            seen_roots.add(root_key)

            root_path = Path(root)
            if not root_path.is_dir():
                continue
            try:
                files = list(root_path.rglob("*.gguf"))
            except Exception:
                continue
            for file_path in files:
                if not file_path.is_file() or "mmproj" in file_path.name.lower():
                    continue
                try:
                    names.add(file_path.relative_to(root_path).as_posix())
                except ValueError:
                    names.add(file_path.name)

    return sorted(names, key=str.lower)


def _resolve_text_encoder_gguf_path(filename: str) -> str | None:
    if not filename or filename.startswith("("):
        return None
    for folder_name in ("text_encoders", "clip_gguf"):
        try:
            full_path = folder_paths.get_full_path(folder_name, filename)
        except Exception:
            full_path = None
        if full_path and os.path.isfile(full_path):
            return os.path.abspath(full_path)
    # Fallback: search models/LLM recursively (get_full_path doesn't cover
    # arbitrary subfolders there the same way).
    for base in get_comfyui_llm_paths():
        if not base.exists():
            continue
        try:
            matches = list(base.glob(f"**/{Path(filename).name}"))
        except Exception:
            continue
        for m in matches:
            if m.is_file() and m.suffix.lower() == ".gguf":
                return str(m.resolve())
    return None


_LORA_FILE_EXTENSIONS = (".gguf", ".safetensors")

# HF module path (relative to a transformer layer) -> llama.cpp GGUF tensor suffix.
_LORA_MODULE_TO_GGUF = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "self_attn.q_norm": "attn_q_norm",
    "self_attn.k_norm": "attn_k_norm",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "mlp.gate": "ffn_gate_inp",
    # Qwen3-Next / Qwen3.5 hybrid linear attention (Gated DeltaNet)
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.out_proj": "ssm_out",
    "linear_attn.conv1d": "ssm_conv1d",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
}

# Non-layer tensors (embeddings / final norm / output head).
_LORA_NON_LAYER_TO_GGUF = {
    "embed_tokens": "token_embd",
    "model.embed_tokens": "token_embd",
    "model.language_model.embed_tokens": "token_embd",
    "lm_head": "output",
    "norm": "output_norm",
    "model.norm": "output_norm",
    "model.language_model.norm": "output_norm",
}


def _list_lora_files() -> list[str]:
    """List LoRA adapters (.gguf and .safetensors) from models/loras (recursively)."""
    names: set[str] = set()
    seen_roots: set[str] = set()
    try:
        roots = folder_paths.get_folder_paths("loras")
    except Exception:
        return []

    for root in roots:
        root_key = os.path.normcase(os.path.abspath(root))
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)

        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for extension in _LORA_FILE_EXTENSIONS:
            try:
                files = list(root_path.rglob(f"*{extension}"))
            except Exception:
                continue
            for file_path in files:
                if not file_path.is_file():
                    continue
                try:
                    names.add(file_path.relative_to(root_path).as_posix())
                except ValueError:
                    names.add(file_path.name)

    return sorted(names, key=str.lower)


def _resolve_lora_file(filename: str) -> str | None:
    if not filename or filename.startswith("("):
        return None
    try:
        full_path = folder_paths.get_full_path("loras", filename)
    except Exception:
        full_path = None
    if full_path and os.path.isfile(full_path):
        return os.path.abspath(full_path)
    return None


def _read_gguf_header(path: Path) -> tuple[dict, dict[str, list[int]]]:
    """Read GGUF metadata and tensor dimensions without loading tensor data."""
    tensors: dict[str, list[int]] = {}
    metadata: dict = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path.name}")
        f.read(4)  # version
        n_tensors, = struct.unpack("<Q", f.read(8))
        n_kv, = struct.unpack("<Q", f.read(8))

        def read_string() -> str:
            length, = struct.unpack("<Q", f.read(8))
            return f.read(length).decode("utf-8", "replace")

        _fixed = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                  6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

        def read_value(value_type: int):
            if value_type in _fixed:
                fmt = _fixed[value_type]
                value = struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
                return bool(value) if value_type == 7 else value
            if value_type == 8:
                return read_string()
            if value_type == 9:
                item_type, = struct.unpack("<I", f.read(4))
                count, = struct.unpack("<Q", f.read(8))
                for _ in range(count):
                    if item_type == 8:
                        read_string()
                    else:
                        read_value(item_type)
                return None
            raise ValueError(f"Unsupported GGUF metadata type: {value_type}")

        for _ in range(n_kv):
            key = read_string()
            value_type, = struct.unpack("<I", f.read(4))
            metadata[key] = read_value(value_type)

        for _ in range(n_tensors):
            name = read_string()
            n_dims, = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            f.read(4)  # dtype
            f.read(8)  # data offset
            tensors[name] = dims

    return metadata, tensors


def _lora_base_tensor_name(tensor_name: str) -> str:
    """Strip PEFT wrappers/LoRA suffixes: `...q_proj.lora_A.weight` -> `...q_proj.weight`."""
    name = re.sub(r"^base_model\.model\.", "", tensor_name)
    name = re.sub(r"\.lora_(A|B)(\.[^.]+)?\.weight$", ".weight", name)
    name = re.sub(r"\.lora_(A|B)(\.[^.]+)?$", ".weight", name)
    name = re.sub(r"\.lora_embedding_(A|B)$", ".weight", name)
    return name


def _lora_target_tensor_name(tensor_name: str) -> str | None:
    """Map an HF LoRA tensor name to the equivalent llama.cpp GGUF tensor name."""
    base_name = _lora_base_tensor_name(tensor_name)
    match = re.search(r"(?:^|\.)layers\.(\d+)\.(.+)$", base_name)
    if match:
        layer_id = int(match.group(1))
        module = match.group(2)
        if module.endswith(".weight"):
            module = module[:-len(".weight")]
        suffix = _LORA_MODULE_TO_GGUF.get(module)
        return f"blk.{layer_id}.{suffix}.weight" if suffix else None

    stem = base_name[:-len(".weight")] if base_name.endswith(".weight") else base_name
    suffix = _LORA_NON_LAYER_TO_GGUF.get(stem)
    return f"{suffix}.weight" if suffix else None


def _gguf_meta_int(metadata: dict, suffix: str) -> int:
    for key, value in metadata.items():
        if key.endswith(suffix) and isinstance(value, int):
            return value
    return 0


def _linear_attn_reorder_index(n: int, num_k_heads: int, num_v_per_k: int, head_dim: int):
    import torch

    return torch.arange(n).reshape(num_k_heads, num_v_per_k, head_dim).permute(1, 0, 2).contiguous().reshape(n)


def _apply_linear_attn_lora_reorder(source_name: str, lora_a, lora_b, metadata: dict):
    """Match llama.cpp's V-head tiled layout for Qwen3-Next/Qwen3.5 linear attention."""
    num_k_heads = _gguf_meta_int(metadata, ".ssm.group_count")
    num_v_heads = _gguf_meta_int(metadata, ".ssm.time_step_rank")
    head_v_dim = _gguf_meta_int(metadata, ".ssm.state_size")
    if num_k_heads <= 0 or num_v_heads <= 0 or head_v_dim <= 0 or num_k_heads == num_v_heads:
        return lora_a, lora_b
    num_v_per_k = num_v_heads // num_k_heads
    if num_k_heads * num_v_per_k != num_v_heads or "linear_attn." not in source_name:
        return lora_a, lora_b

    import torch

    if "linear_attn.in_proj_qkv" in source_name:
        v_dim = num_v_heads * head_v_dim
        k_dim = (lora_b.shape[0] - v_dim) // 2
        if k_dim <= 0 or lora_b.shape[0] != 2 * k_dim + v_dim:
            return lora_a, lora_b
        index = _linear_attn_reorder_index(v_dim, num_k_heads, num_v_per_k, head_v_dim)
        lora_b = torch.cat([lora_b[: 2 * k_dim], lora_b[2 * k_dim:][index]], dim=0)
    elif "linear_attn.in_proj_z" in source_name:
        index = _linear_attn_reorder_index(
            num_v_heads * head_v_dim, num_k_heads, num_v_per_k, head_v_dim
        )
        lora_b = lora_b[index]
    elif "linear_attn.in_proj_a" in source_name or "linear_attn.in_proj_b" in source_name:
        index = _linear_attn_reorder_index(num_v_heads, num_k_heads, num_v_per_k, 1)
        lora_b = lora_b[index]
    elif "linear_attn.out_proj" in source_name:
        index = _linear_attn_reorder_index(
            num_v_heads * head_v_dim, num_k_heads, num_v_per_k, head_v_dim
        )
        lora_a = lora_a[:, index]

    return lora_a, lora_b


def _find_lora_alpha(safetensors_path: Path, fallback_rank: int) -> tuple[float, str | None]:
    candidates = [
        safetensors_path.parent / "adapter_config.json",
        safetensors_path.parent.parent / "adapter_config.json",
    ]
    for config_path in candidates:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            alpha = config.get("lora_alpha")
            if alpha is not None:
                return float(alpha), str(config_path)
        except Exception:
            continue

    try:
        from safetensors import safe_open

        with safe_open(str(safetensors_path), framework="pt") as handle:
            metadata = handle.metadata() or {}
        for key in ("lora_alpha", "alpha"):
            if key in metadata:
                return float(metadata[key]), "safetensors metadata"
    except Exception:
        pass

    return float(fallback_rank), None


def _convert_safetensors_lora_to_gguf(
    safetensors_path: Path,
    base_model_path: Path,
    output_path: Path,
) -> Path:
    """Convert a PEFT .safetensors LoRA to a GGUF adapter matching the base GGUF model.

    This mirrors llama.cpp's convert_lora_to_gguf.py for the tensor names used by
    Qwen VL / Qwen3.5 language models, including the linear-attention V-head reorder.
    """
    try:
        import gguf
        import numpy as np
        import torch
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError(
            "[QwenVL] Auto-converting .safetensors LoRAs requires torch, safetensors and gguf "
            "(the gguf package is installed together with llama-cpp-python)."
        ) from exc

    print(f"[QwenVL] Converting LoRA to GGUF: {safetensors_path.name} -> {output_path.name}")
    metadata, base_tensors = _read_gguf_header(base_model_path)
    architecture = metadata.get("general.architecture")
    if not architecture:
        raise RuntimeError("[QwenVL] Base GGUF has no general.architecture metadata.")

    tensors = load_file(str(safetensors_path), device="cpu")
    pairs: dict[str, dict[str, str]] = {}
    for name in tensors:
        match = re.match(r"(.*)\.lora_(A|B)(?:\.[^.]+)?\.weight$", name)
        if match:
            pairs.setdefault(match.group(1), {})[match.group(2)] = name

    entries = []
    skipped: list[str] = []
    for base_name, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            skipped.append(base_name)
            continue
        target = _lora_target_tensor_name(base_name + ".lora_A.weight")
        if target is None or target not in base_tensors:
            skipped.append(base_name)
            continue

        lora_a = tensors[ab["A"]]
        lora_b = tensors[ab["B"]]
        base_dims = base_tensors[target]
        if (
            len(base_dims) != 2
            or lora_a.shape[1] != base_dims[0]
            or lora_b.shape[0] != base_dims[1]
            or lora_a.shape[0] != lora_b.shape[1]
        ):
            skipped.append(base_name)
            continue

        lora_a = lora_a.to(torch.bfloat16).contiguous()
        lora_b = lora_b.to(torch.bfloat16).contiguous()
        lora_a, lora_b = _apply_linear_attn_lora_reorder(base_name, lora_a, lora_b, metadata)
        if target == "token_embd.weight":
            lora_a = lora_a.T.contiguous()
        entries.append((target, lora_a, lora_b, lora_a.shape[0]))

    if not entries:
        raise RuntimeError(
            f"[QwenVL] No supported LoRA tensors found in {safetensors_path.name}. "
            "Only language-model LoRAs for Qwen VL GGUF models can be converted."
        )

    if skipped:
        examples = ", ".join(skipped[:3])
        print(
            f"[QwenVL] LoRA conversion: skipped {len(skipped)} unsupported tensor(s) "
            f"(e.g. {examples}). Vision-tower LoRAs are not supported."
        )

    alpha, alpha_source = _find_lora_alpha(safetensors_path, entries[0][3])
    if alpha_source is None:
        print(
            f"[QwenVL] LoRA alpha not found (no adapter_config.json); assuming "
            f"alpha={alpha:g} (scale 1.0). Put adapter_config.json next to the "
            "safetensors if your training used a different alpha."
        )
    else:
        print(f"[QwenVL] LoRA alpha={alpha:g} (from {alpha_source}).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")

    writer = gguf.GGUFWriter(str(temp_path), str(architecture))
    writer.add_type(getattr(getattr(gguf, "GGUFType", None), "ADAPTER", "adapter"))
    writer.add_string("adapter.type", "lora")
    writer.add_name(output_path.stem)
    writer.add_float32("adapter.lora.alpha", float(alpha))

    def add_bf16(tensor_name: str, tensor) -> None:
        writer.add_tensor(
            tensor_name,
            tensor.contiguous().view(torch.int16).numpy(),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )

    try:
        for target, lora_a, lora_b, _rank in entries:
            add_bf16(target + ".lora_a", lora_a)
            add_bf16(target + ".lora_b", lora_b)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        os.replace(temp_path, output_path)
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    print(
        f"[QwenVL] LoRA GGUF written: {output_path} "
        f"({len(entries)} tensor pairs, arch={architecture})"
    )
    return output_path


def _ensure_gguf_lora(lora_file: Path, base_model_path: Path) -> Path:
    """Return a GGUF version of the selected LoRA, converting .safetensors if needed."""
    if lora_file.suffix.lower() == ".gguf":
        return lora_file
    if lora_file.suffix.lower() != ".safetensors":
        raise ValueError(f"[QwenVL] Unsupported LoRA format: {lora_file.suffix}")

    target = lora_file.with_suffix(".gguf")
    if target.is_file():
        print(f"[QwenVL] Using existing GGUF LoRA: {target.name}")
        return target

    cache_tag = hashlib.sha1(str(lora_file.resolve()).encode("utf-8")).hexdigest()[:8]
    cache_path = PLUGIN_DIR / "cache" / "loras" / f"{lora_file.stem}-{cache_tag}.gguf"
    if cache_path.is_file():
        print(f"[QwenVL] Using cached GGUF LoRA: {cache_path.name}")
        return cache_path

    try:
        return _convert_safetensors_lora_to_gguf(lora_file, base_model_path, target)
    except OSError as exc:
        # Read-only model folders (or permission issues) -> use the plugin cache.
        print(f"[QwenVL] Could not write next to the LoRA ({exc}); using plugin cache.")
        return _convert_safetensors_lora_to_gguf(lora_file, base_model_path, cache_path)


class QwenVLGGUFBase:
    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
        self.active_loras = None

    def clear(self):
        llm = self.llm
        if llm is not None:
            # llama-cpp-python frees the base model before the LoRA registry in
            # LlamaModel.close(), which can crash on adapters. Unload adapters
            # explicitly while the model is still alive.
            try:
                if int(getattr(llm, "loaded_lora_count", 0) or 0) > 0:
                    llm.unload_all_loras()
            except Exception as exc:
                print(f"[QwenVL] LoRA unload during clear failed: {exc}")
            try:
                close = getattr(llm, "close", None)
                if callable(close):
                    close()
            except Exception as exc:
                print(f"[QwenVL] Llama close during clear failed: {exc}")
        self.llm = None
        self.chat_handler = None
        self.current_signature = None
        self.active_loras = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _install_lora_forwarding(self):
        """Make active LoRA adapters reach the llama.cpp generation internals.

        llama-cpp-python 0.4.x multimodal chat handlers (e.g.
        ``Qwen3VLChatHandler``) accept ``active_loras`` in their ``**kwargs``
        but never forward it to ``Llama.eval`` / ``Llama.create_completion``.
        ``Llama.generate`` then receives ``active_loras=None`` and calls
        ``LlamaContext.clear_loras()``, so an adapter registered through
        ``load_lora()`` is silently ignored for vision prompts. Wrapping the
        bound methods the handler calls internally keeps the adapter attached
        during both prompt prefill and token generation.
        """
        llm = self.llm
        if llm is None or getattr(llm, "_qwenvl_lora_forwarding", False):
            return

        owner = self
        original_eval = llm.eval
        original_create_completion = llm.create_completion

        def eval_with_active_loras(tokens, *args, **kwargs):
            loras = owner.active_loras
            if loras and kwargs.get("active_loras") is None:
                if not args:
                    kwargs["active_loras"] = loras
                elif args[0] is None:
                    # active_loras is the first positional parameter after tokens.
                    args = (loras,) + args[1:]
            return original_eval(tokens, *args, **kwargs)

        def create_completion_with_active_loras(*args, **kwargs):
            if kwargs.get("active_loras") is None and owner.active_loras:
                kwargs["active_loras"] = owner.active_loras
            return original_create_completion(*args, **kwargs)

        try:
            llm.eval = eval_with_active_loras
            llm.create_completion = create_completion_with_active_loras
            llm._qwenvl_lora_forwarding = True
        except Exception as exc:
            print(f"[QwenVL] Warning: could not install LoRA forwarding hooks: {exc}")

    def _prime_lora_context(self):
        """Attach loaded adapters to the llama context before multimodal prefill.

        The MTMD prompt prefill decodes image chunks directly through the native
        context helper, bypassing ``Llama.eval``. Ensuring the adapters are
        attached here means the visual prompt is encoded with the LoRA active
        as well, not just the decoded tokens.
        """
        if not self.active_loras or self.llm is None:
            return
        registry = getattr(getattr(self.llm, "_model", None), "_lora_registry", None)
        if not registry:
            return

        adapters = []
        for entry in self.active_loras:
            name = entry.get("name")
            adapter = registry.get(name)
            if adapter is None:
                print(
                    f"[QwenVL] Warning: LoRA adapter '{name}' is not registered in llama.cpp; "
                    "it will not be applied."
                )
                continue
            adapters.append((adapter, float(entry.get("scale", 1.0))))

        if not adapters:
            return
        try:
            self.llm._ctx.apply_loras(adapters)
        except Exception as exc:
            print(f"[QwenVL] Warning: could not activate LoRA adapter on the llama context: {exc}")

    def _load_backend(self):
        try:
            from llama_cpp import Llama  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "[QwenVL] llama_cpp is not available. Install the GGUF vision dependency first. See docs/GGUF_MANUAL_INSTALL.md"
            ) from exc

    def _load_model(
        self,
        model_name: str,
        device: str,
        ctx: int | None,
        n_batch: int | None,
        gpu_layers: int | None,
        image_max_tokens: int | None,
        top_k: int | None,
        pool_size: int | None,
        model_path=None,
    ):
        self._load_backend()

        custom_model, lora_file, lora_scale = _parse_model_path_payload(model_path)
        if lora_file is not None:
            if not lora_file.is_file():
                raise FileNotFoundError(f"[QwenVL] LoRA adapter not found: {lora_file}")
            lora_scale = max(0.0, float(lora_scale))
            if lora_scale == 0.0:
                print("[QwenVL] LoRA strength is 0.0; the adapter will have no effect.")

        if custom_model is not None:
            # Explicit model path (e.g. a Qwen VL GGUF from models/text_encoders).
            # No catalog lookup or downloading is needed; mmproj is picked up next to it.
            if custom_model.suffix.lower() != ".gguf":
                raise ValueError(
                    f"[QwenVL] The GGUF nodes require a .gguf model file; "
                    f"'{custom_model.name}' is not a GGUF model. Connect this file to "
                    "the QwenVL (transformers) nodes instead."
                )
            if not custom_model.is_file():
                raise FileNotFoundError(f"[QwenVL] Custom GGUF model path not found: {custom_model}")

            model_file = custom_model
            mmproj_file = _find_mmproj_alongside(model_file)

            if mmproj_file is None:
                print(f"[QwenVL] No mmproj found alongside {model_file.name}; running text-only.")
            else:
                print(f"[QwenVL] Using visual projector alongside custom model: {mmproj_file.name}")

            resolved = _resolve_model_entry_for_path(model_file)
            repo_ids: list[str] = []
        else:
            resolved = _resolve_model_entry(model_name)
            base_dir = resolve_base_dir(GGUF_VL_CATALOG.get("base_dir") or "LLM/GGUF")

            author_dir = safe_dirname(resolved.author or "")
            repo_dir = safe_dirname(resolved.repo_dirname)
            target_dir = base_dir / author_dir / repo_dir

            existing_model = find_local_gguf_file(resolved.model_filename, target_dir)
            if existing_model:
                model_file = existing_model
            else:
                model_file = target_dir / Path(resolved.model_filename).name

            existing_mmproj = find_local_gguf_file(resolved.mmproj_filename, target_dir, allow_recursive=False)
            if existing_mmproj:
                mmproj_file = existing_mmproj
            else:
                mmproj_file = target_dir / Path(resolved.mmproj_filename).name if resolved.mmproj_filename else None

            repo_ids = []
            if resolved.repo_id:
                repo_ids.append(resolved.repo_id)
            repo_ids.extend(resolved.alt_repo_ids)

            if not model_file.exists():
                if not repo_ids:
                    raise FileNotFoundError(f"[QwenVL] GGUF model not found locally and no repo_id provided: {model_file}")
                _download_single_file(repo_ids, resolved.model_filename, model_file)

            # Smart mmproj resolution:
            # 1. If configured filename exists in model's directory, use it.
            # 2. If configured filename does not exist, try downloading it from model repo.
            # 3. If no filename configured or download failed, search model's own target_dir locally for *mmproj*.gguf.
            # 4. If still not found, search remote repo_ids for *mmproj*.gguf, download, and use.
            if mmproj_file is not None and not mmproj_file.exists():
                if repo_ids:
                    try:
                        _download_single_file(repo_ids, resolved.mmproj_filename, mmproj_file)
                    except Exception as exc:
                        print(f"[QwenVL] Configured mmproj download failed ({resolved.mmproj_filename}): {exc}")

            if mmproj_file is None or not mmproj_file.exists():
                # Check ONLY model's own target_dir for any matching mmproj (never hijack from other model folders)
                local_mmprojs = []
                if target_dir.exists():
                    local_mmprojs = list(target_dir.glob("*mmproj*.gguf"))
                if local_mmprojs:
                    model_stem = Path(resolved.model_filename).stem.lower()
                    matched_local = None
                    for lm in local_mmprojs:
                        lm_name = lm.name.lower()
                        if "q8" in model_stem and "q8" in lm_name:
                            matched_local = lm
                            break
                        elif "f16" in lm_name or "bf16" in lm_name:
                            if matched_local is None:
                                matched_local = lm
                    mmproj_file = matched_local or local_mmprojs[0]
                    print(f"[QwenVL] Auto-detected local visual projector: {mmproj_file.name}")

            if (mmproj_file is None or not mmproj_file.exists()) and repo_ids:
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    for rid in repo_ids:
                        repo_files = api.list_repo_files(repo_id=rid)
                        mmproj_files = [f for f in repo_files if "mmproj" in f.lower() and f.endswith(".gguf")]
                        if mmproj_files:
                            model_stem = Path(resolved.model_filename).stem.lower()
                            chosen_mmproj = mmproj_files[0]
                            for mf in mmproj_files:
                                mf_lower = mf.lower()
                                if "q8" in model_stem and "q8" in mf_lower:
                                    chosen_mmproj = mf
                                    break
                                elif "f16" in mf_lower or "bf16" in mf_lower:
                                    chosen_mmproj = mf

                            target_mmproj_file = target_dir / Path(chosen_mmproj).name
                            print(f"[QwenVL] Auto-discovering visual projector from {rid}: {chosen_mmproj}")
                            _download_single_file([rid], chosen_mmproj, target_mmproj_file)
                            if target_mmproj_file.exists():
                                mmproj_file = target_mmproj_file
                                print(f"[QwenVL] Successfully auto-downloaded visual projector: {mmproj_file.name}")
                                break
                except Exception as exc:
                    print(f"[QwenVL] Remote mmproj auto-discovery skipped: {exc}")

        # Defensive fallback: accept a raw .safetensors LoRA path in the payload too.
        if lora_file is not None and lora_file.suffix.lower() != ".gguf":
            lora_file = _ensure_gguf_lora(lora_file, model_file)

        device_kind = _pick_device(device)

        n_ctx = int(ctx) if ctx is not None else resolved.context_length
        n_batch_val = int(n_batch) if n_batch is not None else resolved.n_batch
        top_k_val = int(top_k) if top_k is not None else resolved.top_k
        pool_size_val = int(pool_size) if pool_size is not None else resolved.pool_size

        if device_kind == "cuda":
            n_gpu_layers = int(gpu_layers) if gpu_layers is not None else resolved.gpu_layers
        else:
            n_gpu_layers = 0

        img_max = int(image_max_tokens) if image_max_tokens is not None else resolved.image_max_tokens

        has_mmproj = mmproj_file is not None and mmproj_file.exists()

        signature = (
            str(model_file),
            str(mmproj_file) if has_mmproj else "",
            str(lora_file) if lora_file else "",
            round(lora_scale, 4),
            n_ctx,
            n_batch_val,
            n_gpu_layers,
            img_max,
            top_k_val,
            pool_size_val,
        )
        if self.llm is not None and self.current_signature == signature:
            return

        self.clear()

        from llama_cpp import Llama

        self.chat_handler = None
        if has_mmproj:
            handler_cls = None
            try:
                from llama_cpp.llama_chat_format import Qwen3VLChatHandler

                handler_cls = Qwen3VLChatHandler
            except ImportError:
                try:
                    from llama_cpp.llama_chat_format import Qwen25VLChatHandler

                    handler_cls = Qwen25VLChatHandler
                    print(
                        "[QwenVL] Qwen3VLChatHandler is not available in the installed llama-cpp-python; "
                        "falling back to Qwen25VLChatHandler."
                    )
                except ImportError:
                    raise RuntimeError(
                        "[QwenVL] Missing Qwen VL chat handler in llama_cpp. Install the correct fork/wheel. See docs/GGUF_MANUAL_INSTALL.md"
                    )

            # llama-cpp-python >= 0.4 renamed `clip_model_path` to `mmproj_path`,
            # keeping the old name only as a deprecated alias.
            mmproj_key = "mmproj_path" if _handler_supports_param(handler_cls, "mmproj_path") else "clip_model_path"
            mmproj_kwargs = {
                mmproj_key: str(mmproj_file),
                "image_max_tokens": img_max,
                "force_reasoning": False,
                "verbose": False,
            }
            mmproj_kwargs = filter_kwargs_for_callable(getattr(handler_cls, "__init__", handler_cls), mmproj_kwargs)
            if "image_max_tokens" not in mmproj_kwargs:
                print(
                    "[QwenVL] Warning: installed llama_cpp chat handler does not support image_max_tokens; "
                    "image token budget will be controlled by ctx only."
                )
            self.chat_handler = handler_cls(**mmproj_kwargs)

        llm_kwargs = {
            "model_path": str(model_file),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch_val,
            "swa_full": True,
            "verbose": False,
            "pool_size": pool_size_val,
            "top_k": top_k_val,
        }
        if has_mmproj and self.chat_handler is not None:
            llm_kwargs["chat_handler"] = self.chat_handler
            llm_kwargs["image_min_tokens"] = 1024
            llm_kwargs["image_max_tokens"] = img_max

        # Older llama-cpp-python builds accept lora_path in the constructor;
        # 0.4+ uses load_lora() + active_loras during generation instead.
        lora_via_constructor = False
        if lora_file is not None and not hasattr(Llama, "load_lora"):
            try:
                llama_init_params = inspect.signature(Llama.__init__).parameters
            except Exception:
                llama_init_params = {}
            if "lora_path" in llama_init_params:
                llm_kwargs["lora_path"] = str(lora_file)
                lora_via_constructor = True

        print(f"[QwenVL] Loading GGUF: {model_file.name} (device={device_kind}, gpu_layers={n_gpu_layers}, ctx={n_ctx})")
        llm_kwargs_filtered = filter_kwargs_for_callable(getattr(Llama, "__init__", Llama), llm_kwargs)
        if has_mmproj and self.chat_handler is not None and "chat_handler" not in llm_kwargs_filtered:
            print(
                "[QwenVL] Warning: installed llama_cpp Llama() does not accept chat_handler; images will be ignored. "
                "Update llama-cpp-python to a multimodal-capable build."
            )
        if device_kind == "cuda" and n_gpu_layers == 0:
            print("[QwenVL] Warning: device=cuda selected but n_gpu_layers=0; model will run on CPU.")
        self.llm = Llama(**llm_kwargs_filtered)
        self.current_signature = signature
        self.active_loras = None

        if lora_file is not None:
            if hasattr(self.llm, "load_lora"):
                lora_name = "qwenvl_lora"
                try:
                    self.llm.load_lora(lora_name, str(lora_file))
                except Exception as exc:
                    raise RuntimeError(
                        f"[QwenVL] Failed to load LoRA adapter {lora_file.name}: {exc}"
                    ) from exc
                self.active_loras = [{"name": lora_name, "scale": lora_scale}]
                self._install_lora_forwarding()
                print(f"[QwenVL] LoRA adapter loaded: {lora_file.name} (strength {lora_scale:g})")
            elif lora_via_constructor:
                print(f"[QwenVL] LoRA adapter applied via Llama(lora_path=...): {lora_file.name}")
            elif hasattr(self.llm, "apply_lora_from_file"):
                try:
                    self.llm.apply_lora_from_file(str(lora_file))
                except Exception as exc:
                    raise RuntimeError(
                        f"[QwenVL] Failed to apply LoRA adapter {lora_file.name}: {exc}"
                    ) from exc
                print(f"[QwenVL] LoRA adapter applied: {lora_file.name}")
            else:
                print(
                    "[QwenVL] WARNING: installed llama-cpp-python does not support LoRA adapters; "
                    f"ignoring {lora_file.name}."
                )

        # Dynamic ggml backends (CUDA/Metal/etc.) are loaded lazily by Llama(),
        # so GPU support can only be queried reliably after construction.
        if device_kind == "cuda":
            gpu_offload = _llama_supports_gpu_offload()
            if gpu_offload is False:
                print(
                    "[QwenVL] WARNING: CUDA was requested, but llama.cpp loaded no GPU backend. "
                    "The model is running on CPU. Install a CUDA-enabled llama-cpp-python wheel "
                    "(see docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md)."
                )
            elif gpu_offload is True and n_gpu_layers != 0:
                print(f"[QwenVL] llama.cpp GPU offload active (n_gpu_layers={n_gpu_layers}).")

    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: list[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        user_content: list | None = None,
    ) -> str:
        if user_content is not None and self.chat_handler is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        elif images_b64 and self.chat_handler is not None:
            content = [{"type": "text", "text": user_prompt}]
            for img in images_b64:
                if not img:
                    continue
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        else:
            if (images_b64 or user_content is not None) and self.chat_handler is None:
                print("[QwenVL] Warning: Image provided but model has no visual projector (mmproj); running in text-only mode.")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        completion_kwargs = {
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repeat_penalty": float(repetition_penalty),
            "seed": int(seed),
            "stop": ["<|im_end|>", "<|im_start|>"],
        }

        if self.active_loras:
            try:
                supports_active_loras = "active_loras" in inspect.signature(
                    self.llm.create_chat_completion
                ).parameters
            except Exception:
                supports_active_loras = False
            if supports_active_loras:
                completion_kwargs["active_loras"] = self.active_loras
            else:
                print(
                    "[QwenVL] Warning: installed llama-cpp-python does not support active_loras; "
                    "the LoRA adapter will not be applied."
                )

        # Catch a Cancel pressed while the model was loading before starting prefill.
        model_management.throw_exception_if_processing_interrupted()

        # Multimodal chat handlers prefill image chunks through the native
        # context before create_completion() runs, so attach the adapters
        # up-front; the eval()/create_completion() hooks installed alongside
        # load_lora() keep them active for the rest of the request.
        if self.active_loras:
            self._prime_lora_context()

        # Stream tokens so ComfyUI's Cancel button can stop generation. A
        # non-streaming llama.cpp call cannot be interrupted until it finishes.
        start = time.perf_counter()
        completion_kwargs["stream"] = True
        stream = self.llm.create_chat_completion(**completion_kwargs)

        prompt_tokens = None
        completion_tokens = 0
        if isinstance(stream, dict):
            # Some chat handlers ignore stream=True and return a full response.
            result = stream
            usage = result.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            content = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
        else:
            pieces: list[str] = []
            try:
                for chunk in stream:
                    model_management.throw_exception_if_processing_interrupted()
                    choices = chunk.get("choices") or []
                    if choices:
                        piece = (choices[0].get("delta") or {}).get("content")
                        if piece:
                            pieces.append(piece)
                            completion_tokens += 1
            except model_management.InterruptProcessingException:
                print(f"[QwenVL] Generation interrupted after {completion_tokens} token(s).")
                raise
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        print(f"[QwenVL] Stream close failed: {exc}")
            content = "".join(pieces)
            total_tokens = int(getattr(self.llm, "n_tokens", 0) or 0)
            if completion_tokens and total_tokens > completion_tokens:
                prompt_tokens = total_tokens - completion_tokens

        elapsed = max(time.perf_counter() - start, 1e-6)
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            tok_s = completion_tokens / elapsed
            if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                print(
                    f"[QwenVL] Tokens: prompt={prompt_tokens}, completion={completion_tokens}, "
                    f"time={elapsed:.2f}s, speed={tok_s:.2f} tok/s"
                )
            else:
                print(f"[QwenVL] Tokens: completion={completion_tokens}, time={elapsed:.2f}s, speed={tok_s:.2f} tok/s")

        cleaned = clean_model_output(str(content or ""), OutputCleanConfig(mode="text"))
        return cleaned.strip()

    def run(
        self,
        model_name: str,
        preset_prompt: str,
        custom_prompt: str,
        image,
        video,
        frame_count: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        keep_model_loaded: bool,
        device: str,
        ctx: int | None,
        n_batch: int | None,
        gpu_layers: int | None,
        image_max_tokens: int | None,
        top_k: int | None,
        pool_size: int | None,
        video_frame_size: str = "auto",
        model_path=None,
        duration: float = 8.0,
    ):
        torch.manual_seed(int(seed))

        device, gpu_layers, duration = _sanitize_control_values(device, gpu_layers, duration)
        is_fl2va = preset_prompt == FL2VA_PRESET
        is_ref2va = preset_prompt == REF2VA_PRESET
        custom_model, _lora_file, _lora_scale = _parse_model_path_payload(model_path)
        if custom_model is not None:
            resolved = _resolve_model_entry_for_path(custom_model)
        else:
            resolved = _resolve_model_entry(model_name)
        effective_ctx = int(ctx) if ctx is not None else resolved.context_length

        images_b64: list[str] = []
        user_content = None
        load_ctx = ctx

        if is_fl2va or is_ref2va:
            try:
                video_duration = float(duration)
            except Exception:
                video_duration = 8.0
            if not math.isfinite(video_duration) or video_duration <= 0:
                video_duration = 8.0

            min_max_tokens = max(FL2VA_MIN_MAX_TOKENS, REF2VA_MIN_MAX_TOKENS)
            if int(max_tokens) < min_max_tokens:
                print(f"[QwenVL] Prompt Enhance preset: raising max_tokens to {min_max_tokens}.")
                max_tokens = min_max_tokens

            if is_fl2va:
                min_ctx = FL2VA_MIN_CTX
                for frame in _select_fl2va_frames(image, video):
                    img = _tensor_to_base64_png(frame, max_side=1280)
                    if img:
                        images_b64.append(img)
                if len(images_b64) < 2:
                    print(
                        "[QwenVL] FL2VA preset expects two frames (first + last); "
                        f"got {len(images_b64)}. Falling back to "
                        f"{'I2VA' if images_b64 else 'T2VA'} instructions."
                    )
                task_type, prompt, user_content = _build_fl2va_request(images_b64, video_duration, custom_prompt)
                system_prompt = _load_fl2va_system_prompt()
                print(
                    f"[QwenVL] FL2VA preset active: task={task_type}, "
                    f"duration={video_duration:g}s, frames={len(images_b64)}."
                )
            else:
                frame_source = "image"
                frames = _iter_batch_frames(image)
                if not frames and video is not None:
                    frames = _sample_video_frames(video, int(frame_count))
                    frame_source = "video frames"
                    if frames:
                        print("[QwenVL] Ref2VA preset: using video frames as reference pictures.")
                elif image is not None and video is not None:
                    print("[QwenVL] Ref2VA preset: using the image input; video input is ignored in this mode.")
                if len(frames) > REF2VA_MAX_IMAGES:
                    print(
                        f"[QwenVL] Ref2VA preset: using the first {REF2VA_MAX_IMAGES} "
                        f"of {len(frames)} reference frames."
                    )
                    frames = frames[:REF2VA_MAX_IMAGES]
                for frame in frames:
                    img = _tensor_to_base64_png(frame, max_side=REF2VA_IMAGE_MAX_SIDE)
                    if img:
                        images_b64.append(img)
                if not images_b64:
                    raise ValueError(
                        "[QwenVL] Ref2VA preset requires at least one reference image. "
                        "Connect an IMAGE batch (or a video) to the node."
                    )
                min_ctx = min(
                    REF2VA_MAX_CTX,
                    REF2VA_MIN_CTX + max(0, len(images_b64) - 2) * REF2VA_CTX_PER_IMAGE,
                )
                prompt, user_content = _build_ref2va_request(images_b64, video_duration, custom_prompt)
                system_prompt = _load_ref2va_system_prompt()
                print(
                    f"[QwenVL] Ref2VA preset active: duration={video_duration:g}s, "
                    f"references={len(images_b64)} ({frame_source})."
                )

            if effective_ctx < min_ctx:
                print(
                    f"[QwenVL] Prompt Enhance preset: raising ctx to {min_ctx} "
                    "for the guide and reference frames."
                )
                effective_ctx = min_ctx
            load_ctx = effective_ctx
        else:
            prompt = _resolve_preset_text(preset_prompt)
            if custom_prompt and custom_prompt.strip():
                prompt = custom_prompt.strip()
            system_prompt = (
                "You are a helpful vision-language assistant. "
                "Answer directly with the final answer only. No <think> and no reasoning."
            )
            if image is not None:
                img = _tensor_to_base64_png(image, max_side=1280)
                if img:
                    images_b64.append(img)
            if video is not None:
                video_max_side = _resolve_safe_video_max_side(
                    video,
                    frame_count=int(frame_count),
                    ctx=effective_ctx,
                    video_frame_size=video_frame_size,
                )
                for frame in _sample_video_frames(video, int(frame_count)):
                    img = _tensor_to_base64_png(frame, max_side=video_max_side)
                    if img:
                        images_b64.append(img)

        pbar = ProgressBar(3)
        pbar.update_absolute(1, 3, None)

        try:
            self._load_model(
                model_name=model_name,
                device=device,
                ctx=load_ctx,
                n_batch=n_batch,
                gpu_layers=gpu_layers,
                image_max_tokens=image_max_tokens,
                top_k=top_k,
                pool_size=pool_size,
                model_path=model_path,
            )
            pbar.update_absolute(2, 3, None)

            if images_b64 and self.chat_handler is None:
                print("[QwenVL] Warning: images provided but this model entry has no mmproj_file; images will be ignored")
            text = self._invoke(
                system_prompt=system_prompt,
                user_prompt=prompt,
                images_b64=images_b64 if self.chat_handler is not None else [],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                user_content=user_content,
            )
            pbar.update_absolute(3, 3, None)
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()


class AILab_QwenVL_GGUF(QwenVLGGUFBase):
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        catalog = _load_gguf_vl_catalog()
        all_models = catalog.get("models") or {}
        model_keys = sorted(list(all_models.keys())) or ["(edit gguf_models.json)"]
        default_model = model_keys[0]

        prompts = unified_preset_prompts()
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]

        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = ["auto", "cpu", "mps"] + gpu_list

        return {
            "required": {
                "model_name": (model_keys, {"default": default_model, "tooltip": "Ignored when model_path is connected."}),
                "device": (device_options, {"default": "auto", "tooltip": "auto uses CUDA/MPS when available. Requires a CUDA-enabled llama-cpp-python build for actual GPU offload."}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200, "tooltip": "Layers offloaded to GPU (-1 = all). Ignored when device=cpu."}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": f"Select '{FL2VA_PRESET}' to rewrite custom_prompt into a MiniMax-H3 FL2VA prompt using the first and last frames from the image (or video) input, or '{REF2VA_PRESET}' to rewrite it into a six-section Ref2VA prompt using all frames of the image batch as <Picture N> references."}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": f"Preset instruction override. For '{FL2VA_PRESET}' and '{REF2VA_PRESET}' this is the original prompt to enhance."}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 8192}),
                # control_after_generate must be declared explicitly: the frontend
                # adds a serialized control widget after every "seed" input, and
                # without this flag its value-restore migration miscounts widget
                # slots when the optional model_path forceInput is present, which
                # shifts device/gpu_layers/duration values on workflow reload.
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "control_after_generate": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
                "model_path": ("STRING", {"forceInput": True, "tooltip": "Connect MODEL_PATH from the QwenVL Path Loader (carries model + optional LoRA), or a plain GGUF path string. Overrides model_name unless the loader's override_model is False."}),
                "duration (FL2VA-Ref2VA)": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 3600.0, "step": 0.1, "tooltip": f"Target video duration in seconds. Used by '{FL2VA_PRESET}' for the last-frame timestamp and by '{REF2VA_PRESET}' to plan shot timing; ignored by other presets."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    def process(
        self,
        model_name,
        device,
        keep_model_loaded,
        gpu_layers,
        preset_prompt,
        custom_prompt,
        max_tokens,
        seed,
        image=None,
        video=None,
        model_path=None,
        **kwargs,
    ):
        duration = _get_duration_input(kwargs)
        return self.run(
            model_name=model_name,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image=image,
            video=video,
            frame_count=16,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            device=device,
            ctx=None,
            n_batch=None,
            gpu_layers=gpu_layers,
            image_max_tokens=None,
            top_k=None,
            pool_size=None,
            video_frame_size="auto",
            model_path=model_path,
            duration=duration,
        )


class AILab_QwenVL_GGUF_Advanced(QwenVLGGUFBase):
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        catalog = _load_gguf_vl_catalog()
        all_models = catalog.get("models") or {}
        model_keys = sorted(list(all_models.keys())) or ["(edit gguf_models.json)"]
        default_model = model_keys[0]

        prompts = unified_preset_prompts()
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]

        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = ["auto", "cpu", "mps"] + gpu_list

        return {
            "required": {
                "model_name": (model_keys, {"default": default_model, "tooltip": "Ignored when model_path is connected."}),
                "device": (device_options, {"default": "auto", "tooltip": "auto uses CUDA/MPS when available. Requires a CUDA-enabled llama-cpp-python build for actual GPU offload."}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200, "tooltip": "Layers offloaded to GPU (-1 = all). Ignored when device=cpu."}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": f"Select '{FL2VA_PRESET}' to rewrite custom_prompt into a MiniMax-H3 FL2VA prompt using the first and last frames from the image (or video) input, or '{REF2VA_PRESET}' to rewrite it into a six-section Ref2VA prompt using all frames of the image batch as <Picture N> references."}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": f"Preset instruction override. For '{FL2VA_PRESET}' and '{REF2VA_PRESET}' this is the original prompt to enhance."}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 16384}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64}),
                "video_frame_size": (["auto", "384", "448", "512", "768", "original"], {"default": "auto"}),
                "ctx": ("INT", {"default": 8192, "min": 1024, "max": 262144, "step": 512}),
                "n_batch": ("INT", {"default": 512, "min": 64, "max": 32768, "step": 64}),
                "image_max_tokens": ("INT", {"default": 4096, "min": 256, "max": 1024000, "step": 256}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 32768}),
                "pool_size": ("INT", {"default": 4194304, "min": 1048576, "max": 10485760, "step": 524288}),
                # Explicit flag keeps the extra control widget in the frontend's
                # widget-value restore mapping; otherwise saved values shift by
                # one slot when model_path's forceInput is present.
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "control_after_generate": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
                "model_path": ("STRING", {"forceInput": True, "tooltip": "Connect MODEL_PATH from the QwenVL Path Loader (carries model + optional LoRA), or a plain GGUF path string. Overrides model_name unless the loader's override_model is False."}),
                "duration (FL2VA-Ref2VA)": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 3600.0, "step": 0.1, "tooltip": f"Target video duration in seconds. Used by '{FL2VA_PRESET}' for the last-frame timestamp and by '{REF2VA_PRESET}' to plan shot timing; ignored by other presets."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    def process(
        self,
        model_name,
        device,
        keep_model_loaded,
        gpu_layers,
        preset_prompt,
        custom_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        frame_count,
        video_frame_size,
        ctx,
        n_batch,
        image_max_tokens,
        top_k,
        pool_size,
        seed,
        image=None,
        video=None,
        model_path=None,
        **kwargs,
    ):
        duration = _get_duration_input(kwargs)
        return self.run(
            model_name=model_name,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image=image,
            video=video,
            frame_count=frame_count,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            keep_model_loaded=keep_model_loaded,
            device=device,
            ctx=ctx,
            n_batch=n_batch,
            gpu_layers=gpu_layers,
            image_max_tokens=image_max_tokens,
            top_k=top_k,
            pool_size=pool_size,
            video_frame_size=video_frame_size,
            model_path=model_path,
            duration=duration,
        )


def _list_hf_model_entries() -> list[str]:
    """List transformers-model entries for the Path Loader.

    Collects .safetensors weights and snapshot folders (dirs containing a
    config.json) from models/LLM, models/text_encoders and models/clip.
    Folder entries are reported with a trailing slash.
    """
    names: set[str] = set()
    seen_roots: set[str] = set()
    for folder_name in ("LLM", "text_encoders", "clip"):
        try:
            roots = folder_paths.get_folder_paths(folder_name)
        except Exception:
            continue
        for root in roots:
            root_key = os.path.normcase(os.path.abspath(root))
            if root_key in seen_roots:
                continue
            seen_roots.add(root_key)

            root_path = Path(root)
            if not root_path.is_dir():
                continue
            try:
                for file_path in root_path.rglob("*.safetensors"):
                    if file_path.is_file():
                        names.add(file_path.relative_to(root_path).as_posix())
                for config_path in root_path.rglob("config.json"):
                    entry_dir = config_path.parent
                    rel = entry_dir.relative_to(root_path).as_posix()
                    if rel and rel != ".":
                        names.add(rel + "/")
            except Exception:
                continue

    return sorted(names, key=str.lower)


def _resolve_loader_model_path(filename: str) -> str | None:
    """Resolve a Path Loader model_file entry to an absolute path.

    GGUF entries are looked up in models/text_encoders (and models/clip);
    safetensors files and snapshot folders in models/LLM, models/text_encoders
    and models/clip. Entries ending in "/" refer to folders.
    """
    if not filename or filename.startswith("("):
        return None

    gguf_path = _resolve_text_encoder_gguf_path(filename)
    if gguf_path:
        return gguf_path

    wants_dir = filename.endswith("/")
    rel = filename.rstrip("/")
    for folder_name in ("LLM", "text_encoders", "clip"):
        try:
            roots = folder_paths.get_folder_paths(folder_name)
        except Exception:
            continue
        for root in roots:
            candidate = Path(root) / rel
            if wants_dir and candidate.is_dir():
                return str(candidate.resolve())
            if not wants_dir and candidate.is_file():
                return str(candidate.resolve())
    return None


class AILab_QwenVL_GGUF_PathLoader:
    """Lightweight path picker shared by every QwenVL node.

    Lists GGUF vision models (models/text_encoders, models/clip) and
    transformers models (.safetensors weights or snapshot folders in
    models/LLM, models/text_encoders, models/clip). Unlike CLIPLoader nodes it
    does not load anything into memory. The single MODEL_PATH output carries a
    dict with the model path + format, the LoRA toggle and the resolved LoRA
    path; each consuming node applies the model with its own backend (llama.cpp
    or transformers) and handles LoRA conversion itself.
    """

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        gguf_files = _list_text_encoder_gguf_files()
        hf_files = _list_hf_model_entries()
        options = gguf_files + hf_files or ["(no models found in models/text_encoders or models/LLM)"]
        lora_files = _list_lora_files()
        lora_options = lora_files or ["(no LoRA found in models/loras)"]
        return {
            "required": {
                "override_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True: connected nodes use the model selected here (plus LoRA if use_lora is on). "
                               "False: MODEL_PATH is ignored downstream and nodes fall back to their own "
                               "model_name dropdown (LoRA is not applied).",
                }),
                "model_file": (options, {
                    "default": options[0],
                    "tooltip": "GGUF files from models/text_encoders (and models/clip), plus .safetensors "
                               "weights and model folders (with config.json) from models/LLM, "
                               "models/text_encoders and models/clip. GGUF entries feed the GGUF nodes, "
                               "safetensors/folder entries feed the transformers nodes.",
                }),
                "use_lora": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable to apply a LoRA adapter on top of the model. "
                               "Only effective when override_model is True.",
                }),
                "lora_file": (lora_options, {
                    "default": lora_options[0],
                    "tooltip": "LoRA adapters discovered in models/loras (recursively). "
                               "GGUF nodes auto-convert .safetensors adapters to GGUF; transformers "
                               "nodes apply them via PEFT (requires the adapter_config.json).",
                }),
                "lora_str": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.0,
                    "max": 8.0,
                    "step": 0.05,
                    "tooltip": "LoRA strength multiplier applied when merging the adapter (0 = disabled).",
                }),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("MODEL_PATH",)
    FUNCTION = "get_paths"
    CATEGORY = "🧪AILab/QwenVL"

    def get_paths(self, override_model=True, model_file=None, use_lora=False, lora_file=None, lora_str=1.0):
        # When override_model is False the MODEL_PATH output is flagged so
        # downstream nodes ignore it and use their own model_name dropdown
        # instead. The LoRA selection is skipped as well, since it only makes
        # sense on top of the model chosen here.
        if not override_model:
            print("[QwenVL] Path Loader: override_model=False, downstream nodes will use their own model_name")
            return ({
                "model_path": "",
                "model_format": "",
                "use_lora": False,
                "lora_path": "",
                "lora_format": "",
                "lora_scale": 1.0,
                "override_model": False,
            },)

        if model_file is None:
            raise FileNotFoundError("[QwenVL] Path Loader: no model_file selected")

        model_path_str = _resolve_loader_model_path(model_file)
        if model_path_str is None:
            raise FileNotFoundError(
                f"[QwenVL] Model file not found in models/text_encoders, models/clip or models/LLM: {model_file}"
            )

        model_path = Path(model_path_str)
        if model_path.is_dir():
            model_format = "hf_dir"
        elif model_path.suffix.lower() == ".gguf":
            model_format = "gguf"
        else:
            model_format = "safetensors"

        lora_path = ""
        lora_format = ""
        try:
            lora_scale = max(0.0, float(lora_str))
        except (TypeError, ValueError):
            lora_scale = 1.0
        if use_lora:
            selected = _resolve_lora_file(lora_file)
            if selected is None:
                raise FileNotFoundError(
                    f"[QwenVL] LoRA file not found in models/loras: {lora_file}"
                )
            lora_path = selected
            lora_format = "gguf" if Path(selected).suffix.lower() == ".gguf" else "safetensors"
            # No conversion here: GGUF nodes convert .safetensors adapters to
            # GGUF themselves; transformers nodes apply them via PEFT.
            print(f"[QwenVL] Path Loader: {model_path.name} + LoRA {Path(lora_path).name} (strength {lora_scale:g})")
        else:
            print(f"[QwenVL] Path Loader: {model_path.name} ({model_format})")

        payload = {
            "model_path": str(model_path),
            "model_format": model_format,
            "use_lora": bool(use_lora),
            "lora_path": lora_path,
            "lora_format": lora_format,
            "lora_scale": lora_scale,
            "override_model": True,
        }
        return (payload,)


NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL_GGUF": AILab_QwenVL_GGUF,
    "AILab_QwenVL_GGUF_Advanced": AILab_QwenVL_GGUF_Advanced,
    "AILab_QwenVL_GGUF_PathLoader": AILab_QwenVL_GGUF_PathLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL_GGUF": "QwenVL (GGUF)",
    "AILab_QwenVL_GGUF_Advanced": "QwenVL Advanced (GGUF)",
    "AILab_QwenVL_GGUF_PathLoader": "QwenVL Path Loader",
}
