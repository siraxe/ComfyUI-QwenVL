# ComfyUI-QwenVL (GGUF)
# User-defined prompt presets for the QwenVL nodes.
#
# Stores user presets in a JSON file next to the built-in system_prompts.json
# and exposes two small HTTP routes consumed by web/js/prompt_presets.js:
#   GET  /qwenvl/prompt_presets  -> {"defaults": {...}, "user": {...}}
#   POST /qwenvl/prompt_presets  -> body {"user": {...}}, returns the saved map
#
# This integration script follows GPL-3.0 License.

import json
import threading

from AILab_Utils import PLUGIN_DIR, SYSTEM_PROMPTS_PATH, load_system_prompts

PRESETS_FILE = PLUGIN_DIR / "AILab_QwenVL_User_Presets.json"

_write_lock = threading.Lock()


def load_default_presets() -> dict[str, str]:
    """Built-in presets (name -> system prompt) shipped in system_prompts.json."""
    try:
        return dict(load_system_prompts().get("qwenvl_prompts") or {})
    except Exception as exc:
        print(f"[QwenVL] Default prompt presets load failed: {exc}")
        return {}


def load_user_presets() -> dict[str, str]:
    """User presets (name -> system prompt) stored in AILab_QwenVL_User_Presets.json."""
    if not PRESETS_FILE.exists():
        return {}
    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"[QwenVL] User prompt presets load failed ({PRESETS_FILE}): {exc}")
        return {}
    if not isinstance(data, dict):
        return {}
    presets: dict[str, str] = {}
    for name, text in data.items():
        name = str(name).strip()
        if name:
            presets[name] = str(text if text is not None else "")
    return presets


def save_user_presets(presets) -> dict[str, str]:
    """Persist user presets atomically; returns the cleaned map that was saved."""
    cleaned: dict[str, str] = {}
    for name, text in (presets or {}).items():
        name = str(name).strip()
        if not name:
            continue
        cleaned[name] = str(text if text is not None else "")

    with _write_lock:
        PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = PRESETS_FILE.with_name(PRESETS_FILE.name + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(cleaned, fh, ensure_ascii=False, indent=2)
            tmp_path.replace(PRESETS_FILE)
        except Exception as exc:
            print(f"[QwenVL] User prompt presets save failed: {exc}")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
    return cleaned


def register_routes():
    """Attach the preset HTTP routes to the running ComfyUI server."""
    try:
        from server import PromptServer
        from aiohttp import web
    except Exception as exc:  # pragma: no cover - depends on host app
        print(f"[QwenVL] Prompt preset routes unavailable: {exc}")
        return

    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return

    # This module can be imported twice (as py.AILab_QwenVL_Presets by the
    # node loader and top-level by the GGUF nodes, since py/ is on sys.path).
    # Registering the same GET route twice makes aiohttp raise a duplicate
    # HEAD route error that aborts server startup, so guard with a flag.
    if getattr(instance, "_qwenvl_prompt_presets_registered", False):
        return
    instance._qwenvl_prompt_presets_registered = True

    async def _get_presets(request):
        return web.json_response(
            {
                "defaults": load_default_presets(),
                "user": load_user_presets(),
            }
        )

    async def _post_presets(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body."}, status=400)
        saved = save_user_presets((payload or {}).get("user"))
        return web.json_response({"user": saved})

    try:
        instance.routes.get("/qwenvl/prompt_presets")(_get_presets)
        instance.routes.post("/qwenvl/prompt_presets")(_post_presets)
    except Exception as exc:
        print(f"[QwenVL] Prompt preset routes registration failed: {exc}")


register_routes()
