# ComfyUI-QwenVL
# Unified prompt-preset system shared by the transformers (HF) and GGUF nodes.
#
# Holds the FL2VA / Ref2VA preset builders, the preset list/resolution used by
# every QwenVL node's preset_prompt widget, and the shared duration widget
# reader. Keeping this in one module guarantees both backends build identical
# prompts for the same preset.
#
# This integration script follows GPL-3.0 License.
#
# Source: https://github.com/1038lab/ComfyUI-QwenVL

import math
import re

from AILab_Utils import PLUGIN_DIR, load_system_prompts
from AILab_QwenVL_Presets import load_user_presets

_prompts = load_system_prompts()
PRESET_PROMPTS = _prompts["preset_prompts"]
SYSTEM_PROMPTS = _prompts["qwenvl_prompts"]

# ------------------------------------------------------------------
# FL2VA preset (MiniMax-H3 first/last frame -> video+audio prompt)
# ------------------------------------------------------------------

FL2VA_PRESET = "🎬 Prompt Enhance FL2VA"
FL2VA_GUIDE_PATH = PLUGIN_DIR / "AILab_FL2VA_Prompt_Guide.md"
FL2VA_MIN_CTX = 16384
FL2VA_MIN_MAX_TOKENS = 4096

_FL2VA_SYSTEM_PREFIX = (
    "You are a professional prompt enhancement engineer for video and audio generation, "
    "with expertise in visual storytelling, shot planning, camera movement, and sound design. "
    "Before enhancing the prompt, carefully interpret the user's intended scene, subjects, actions, and audiovisual requirements. "
    "Preserve the core meaning, event sequence, and explicit constraints. "
    "Add relevant visual and sound details that enrich the intended scene without changing its premise or introducing unrelated events. "
    "Rewrite the user's prompt according to the following guide. "
    "Write the entire enhanced prompt in English, regardless of the original prompt's language. "
    "Apply only the specified task type and return only the final prompt.\n\n"
)

_FL2VA_TASKS = {
    "T2VA": {"roles": (), "instruction": ""},
    "I2VA": {
        "roles": ("first",),
        "instruction": (
            "Inspect the attached <Picture 1>, which is the video's first frame, "
            "and enhance the prompt following the guide's I2VA rules."
        ),
    },
    "FL2VA": {
        "roles": ("first", "last"),
        "instruction": (
            "Inspect the attached <Picture 1> (the video's first frame) and "
            "<Picture 2> (the video's last frame), and enhance the prompt "
            "following the guide's FL2VA rules."
        ),
    },
}


# ------------------------------------------------------------------
# Ref2VA preset (all image-batch frames -> six-section reference prompt)
# ------------------------------------------------------------------

REF2VA_PRESET = "🎬 Prompt Enhance Ref2VA"
REF2VA_GUIDE_PATH = PLUGIN_DIR / "AILab_Ref2VA_Prompt_Guide.md"
REF2VA_MIN_CTX = 16384
REF2VA_MAX_CTX = 32768
REF2VA_CTX_PER_IMAGE = 2048
REF2VA_MAX_IMAGES = 16
REF2VA_IMAGE_MAX_SIDE = 1024
REF2VA_MIN_MAX_TOKENS = 4096

_REF2VA_SYSTEM_PREFIX = (
    "You enhance prompts for reference-conditioned video and audio generation (Ref2VA). "
    "Infer reference roles from the user's prompt and supplied media. "
    "Preserve intent and explicit constraints while adding relevant visual and sound details. "
    "Follow the guide below and return only its six sections in order, "
    "with plain-text headers on separate lines and no Markdown fences.\n\n"
)

_REF2VA_LABEL_NAMES = {
    "zh": {"Subject": "主体", "Picture": "图片", "Video": "视频", "Audio": "音频"},
    "en": {"Subject": "subject ", "Picture": "image ", "Video": "video ", "Audio": "audio "},
}


# ------------------------------------------------------------------
# Unified preset list / resolution
# ------------------------------------------------------------------

def unified_preset_prompts():
    """Preset list shared by every QwenVL node (HF and GGUF).

    Built-ins from system_prompts.json, the two Prompt Enhance presets, and the
    user presets created via the "Edit preset prompts" dialog.
    """
    prompts = list(PRESET_PROMPTS or ["🖼️ Detailed Description"])
    for preset in (FL2VA_PRESET, REF2VA_PRESET):
        if preset not in prompts:
            prompts.append(preset)
    try:
        for name in load_user_presets():
            if name not in prompts:
                prompts.append(name)
    except Exception as exc:
        print(f"[QwenVL] User preset lookup failed: {exc}")
    return prompts


def resolve_preset_text(name: str) -> str:
    """Resolve a preset widget value to its system prompt text.

    Built-ins come from system_prompts.json; user presets from the user preset
    file (read live so the editor dialog takes effect without a restart).
    Anything else is treated as a literal prompt.
    """
    if name in SYSTEM_PROMPTS:
        return SYSTEM_PROMPTS[name]
    try:
        user_presets = load_user_presets()
    except Exception:
        user_presets = {}
    if name in user_presets:
        return user_presets[name]
    return name


# ------------------------------------------------------------------
# FL2VA / Ref2VA request builders
# ------------------------------------------------------------------

def load_fl2va_system_prompt() -> str:
    guide = ""
    try:
        guide = FL2VA_GUIDE_PATH.read_text(encoding="utf-8").strip()
    except Exception as exc:
        print(f"[QwenVL] FL2VA prompt guide load failed ({FL2VA_GUIDE_PATH}): {exc}")
    if not guide:
        guide = (
            "Rewrite the prompt as a MiniMax-H3 audiovisual timeline using an "
            "integrated_multimodal_description, overall_soundscape and non_diegetic_music."
        )
    return _FL2VA_SYSTEM_PREFIX + guide


def select_fl2va_frames(image, video):
    """Pick the first and last reference frames from an IMAGE batch / video tensor."""
    source = image if image is not None else video
    if source is None:
        return []
    if not hasattr(source, "shape"):
        return [source]
    ndim = int(getattr(source, "ndim", 0))
    if ndim == 4:
        count = int(source.shape[0])
        if count <= 0:
            return []
        if count == 1:
            return [source[0]]
        return [source[0], source[-1]]
    if ndim == 3:
        return [source]
    return []


def build_fl2va_request(images_b64, duration: float, custom_prompt: str):
    """Replicate MiniMax-H3-FL2VA request layout (system prefix + guide handled separately)."""
    count = len(images_b64)
    if count >= 2:
        task_type = "FL2VA"
    elif count == 1:
        task_type = "I2VA"
    else:
        task_type = "T2VA"

    task = _FL2VA_TASKS[task_type]
    roles = task["roles"]
    images_b64 = images_b64[: len(roles)]

    original_prompt = (custom_prompt or "").strip()
    if not original_prompt:
        original_prompt = "Describe the motion and audiovisual content that connects the provided reference frames."
        print("[QwenVL] FL2VA preset: custom_prompt is empty; using a generic original prompt.")

    instruction = task["instruction"] + "\n" if roles else ""
    duration_number = int(duration) if float(duration).is_integer() else float(duration)
    header = (
        f"Task type: {task_type}.\n"
        f"Video duration: {duration_number} seconds.\n"
        f"{instruction}"
        "Choose shots that best express the original intent, using separate shots for distinct subjects, details, reactions, or meaningful changes in scene, viewpoint, or time.\n"
        "Allocate shot durations according to content and pacing.\n"
        "Respect explicit shot requests and follow the guide's task-specific, formatting, and transition rules.\n"
        f"Original prompt: {original_prompt}\n"
    )

    user_prompt = header
    content_parts = [{"type": "text", "text": header}]
    for number, (role, image_b64) in enumerate(zip(roles, images_b64), 1):
        timestamp = 0.0 if role == "first" else float(duration)
        line = f"<Picture {number}>: the video's {role} frame at {timestamp:.2f} seconds.\n"
        user_prompt += line
        content_parts.append({"type": "text", "text": line})
        content_parts.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        )

    return task_type, user_prompt, content_parts


def load_ref2va_system_prompt() -> str:
    guide = ""
    try:
        guide = REF2VA_GUIDE_PATH.read_text(encoding="utf-8").strip()
    except Exception as exc:
        print(f"[QwenVL] Ref2VA prompt guide load failed ({REF2VA_GUIDE_PATH}): {exc}")
    if not guide:
        guide = (
            "Rewrite the prompt as a MiniMax-H3 full-reference prompt with the six sections "
            "subject_definitions, summary, retention_analysis, detailed_description, "
            "overall_soundscape and non_diegetic_music."
        )
    return _REF2VA_SYSTEM_PREFIX + guide


def iter_batch_frames(source):
    """Return every frame of an IMAGE batch as a list of single-frame tensors."""
    if source is None:
        return []
    if not hasattr(source, "shape"):
        return [source]
    ndim = int(getattr(source, "ndim", 0))
    if ndim == 4:
        return [source[i] for i in range(int(source.shape[0]))]
    if ndim == 3:
        return [source]
    return []


def build_ref2va_request(images_b64, duration: float, custom_prompt: str):
    """Replicate MiniMax-H3-Ref2VA request layout (system prefix + guide handled separately)."""
    count = len(images_b64)
    original_prompt = (custom_prompt or "").strip()
    if not original_prompt:
        original_prompt = "Use the provided reference images to create the target video."
        print("[QwenVL] Ref2VA preset: custom_prompt is empty; using a generic original prompt.")

    original_prompt = re.sub(r"<Figure\s+(\d+)>", r"<Picture \1>", original_prompt)
    used_pictures = {int(number) for number in re.findall(r"<Picture\s+(\d+)>", original_prompt)}
    if used_pictures and max(used_pictures) > count:
        raise ValueError(
            f"[QwenVL] Ref2VA preset: prompt references <Picture {max(used_pictures)}> "
            f"but only {count} reference image(s) are connected."
        )
    if re.search(r"<(?:Video|Audio)\s+\d+>", original_prompt):
        print(
            "[QwenVL] Ref2VA preset: <Video N>/<Audio N> labels are present, but this node "
            "only supplies image references; those relationships may be approximated by the model."
        )

    names = _REF2VA_LABEL_NAMES["zh"] if re.search("[㐀-鿿]", original_prompt) else _REF2VA_LABEL_NAMES["en"]
    original_prompt = re.sub(
        r"<(Subject|Picture|Video|Audio)\s+(\d+)>",
        lambda match: names[match[1]] + match[2],
        original_prompt,
    )

    user_prompt = f"Target video duration: {float(duration):.6f} seconds.\n"
    if count:
        user_prompt += "Inspect the reference images labeled <Picture N> and use them according to the user's instructions.\n"
    user_prompt += (
        "Choose shots that best express the original intent. "
        "Allocate shot durations and cut timing within the target duration according to content and pacing. "
        "Respect explicit shot requests, reference constraints and the guide's transition rules.\n"
        f"Original prompt:\n{original_prompt}\n"
    )

    content_parts = [{"type": "text", "text": user_prompt}]
    for index, image_b64 in enumerate(images_b64, 1):
        content_parts.append(
            {"type": "text", "text": f"图片{index} / reference image {index}: <Picture {index}>.\n"}
        )
        content_parts.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        )

    return user_prompt, content_parts


def get_duration_input(kwargs: dict, default: float = 8.0) -> float:
    """Read the duration widget, tolerating the key used by older node layouts."""
    for key in ("duration (FL2VA-Ref2VA)", "duration (FL2VA)", "duration"):
        if key in kwargs:
            value = kwargs[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                candidate = float(value)
                if math.isfinite(candidate) and candidate > 0:
                    return candidate
            print(f"[QwenVL] Warning: invalid duration value {value!r}; using {default}.")
            return float(default)
    return float(default)
