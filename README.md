# **QwenVL for ComfyUI**

<table>
  <tr>
    <td width="40%" valign="top">
      <strong>📝 NOTE</strong><br><br>
      <strong>This is a fork of the original <a href="https://github.com/1038lab/ComfyUI-QwenVL">ComfyUI-QwenVL</a> by 1038lab</strong>, extended with additional features such as the Path Loader <code>override_model</code> toggle, LoRA strength (<code>lora_str</code>) control, unified prompt presets across backends, and MiniMax-H3 FL2VA/Ref2VA prompt-enhance presets on both the GGUF and Transformers nodes. All credit for the original project goes to its authors.
    </td>
    <td width="60%" valign="top"><em><img width="843" height="942" alt="vw" src="https://github.com/user-attachments/assets/5abf2ebd-cf40-4d57-8d17-8bb26e090f2b" />
</em></td>
  </tr>
</table>

### 📂 QwenVL Path Loader

**Example workflow**: [`example_workflows/Q3.8_27B_H3_promt_enhance.json`](example_workflows/Q3.8_27B_H3_promt_enhance.json)

**Models / LoRAs used in the workflow (HuggingFace)**:

- GGUF model: any Qwen3.8 27B VL model works
- Prompt-rewriter LoRAs: [siraxe/lightx2v_MiniMax-H3-Prompt-Rewriter_loras](https://huggingface.co/siraxe/lightx2v_MiniMax-H3-Prompt-Rewriter_loras)

```
ComfyUI/
└─ models/
   ├─ LLM/                        <-- GGUF and Transformers models both work here
   │  └─ Q3.8_27B-Q4/
   │     ├─ Q3.8_27B-Q4.gguf
   │     └─ mmproj-model-bf16.gguf   <-- required for image/video input
   └─ loras/
      ├─ lightx2v_minimax_h3_FL2VA_PR-Q3.8-27B_rank64_bf16.safetensors   <-- FL2VA preset
      └─ lightx2v_minimax_h3_Ref2VA_PR-Q3.8-27B_rank64_bf16.safetensors  <-- Ref2VA preset
```

*(GGUF models may alternatively live in `models/text_encoders/` — same layout, also scanned. The model file may also be split into `-00001-of-...` shards, and any `*.gguf` file with `mmproj` in its name is detected as projector — `mmproj-model.gguf` works too, though the loader prefers `mmproj-model-bf16.gguf` when present.)*

The **QwenVL Path Loader** lets you use any local model without touching `custom_models.json`:

1. Place the GGUF model inside `ComfyUI/models/LLM/` **or** `ComfyUI/models/text_encoders/` — **each model in its own folder** (subfolders are scanned recursively). The `mmproj` visual projector must sit **next to the model file in the same folder**. Transformers models (`.safetensors` weights or model folders with `config.json`) are also listed from `models/LLM`, `models/text_encoders` and `models/clip`.
2. Pick the model in the loader's `model_file` dropdown (`mmproj` files are hidden from the list automatically).
3. Optional: enable `use_lora` and pick a LoRA from `models/loras` (`.gguf` or `.safetensors`), and set its strength with `lora_str`. GGUF nodes auto-convert `.safetensors` adapters to GGUF on first use; the Transformers nodes apply them via PEFT (requires `adapter_config.json`).
4. Connect `MODEL_PATH` to any QwenVL node (GGUF or Transformers). Set `override_model` to `False` to temporarily ignore the loader and use the node's own `model_name` dropdown instead.

#### 🎬 MiniMax-H3 Prompt-Rewriter LoRAs

Ready-made LoRAs for the `🎬 Prompt Enhance FL2VA` / `🎬 Prompt Enhance Ref2VA` presets (trained for the **Qwen3.8-27B** prompt rewriter used in the example workflow above) are available from
[siraxe/lightx2v_MiniMax-H3-Prompt-Rewriter_loras](https://huggingface.co/siraxe/lightx2v_MiniMax-H3-Prompt-Rewriter_loras):

| File | Use with preset |
|---|---|
| `lightx2v_minimax_h3_FL2VA_PR-Q3.8-27B_rank64_bf16.safetensors` | `🎬 Prompt Enhance FL2VA` |
| `lightx2v_minimax_h3_Ref2VA_PR-Q3.8-27B_rank64_bf16.safetensors` | `🎬 Prompt Enhance Ref2VA` |

Only one adapter can be attached at a time, so switch `lora_file` when you switch presets (or run one loader per preset, since `override_model=False` lets the node fall back to its own dropdown).



## **🚀 Installation**

1. Clone this repository to your ComfyUI/custom\_nodes directory:  
   ```
   cd ComfyUI/custom\_nodes  
   git clone https://github.com/1038lab/ComfyUI-QwenVL.git
   ```
2. Install the required dependencies:  
   ```
   cd ComfyUI/custom\_nodes/ComfyUI-QwenVL  
   pip install \-r requirements.txt
   ```

3. Restart ComfyUI.

### **Optional: SageAttention Support**
For optimal performance on supported GPUs, install SageAttention:
```
pip install sageattention
```

## **🧭 Node Overview**

### **Transformers (HF) Nodes**
- **QwenVL**: Quick vision-language inference (image/video + preset/custom prompts).  
- **QwenVL (Advanced)**: Full control over sampling, device, and performance settings.  
- **QwenVL Prompt Enhancer**: Text-only prompt enhancement (supports both Qwen3 text models and QwenVL models in text mode).  

### **Utilities Nodes**
- **HuggingFace Downloader**: Directly download GGUF/HF models or entire repositories from HuggingFace to your ComfyUI directories.

### **GGUF (llama.cpp) Nodes**
- **QwenVL (GGUF)**: GGUF vision-language inference.  
- **QwenVL (GGUF Advanced)**: Extended GGUF controls (context, GPU layers, etc.).  
- **QwenVL Path Loader**: Picks a `.gguf` from `models/LLM` or `models/text_encoders` (the matching `mmproj` is auto-detected next to it), or a Transformers model (`.safetensors` weights / model folders with `config.json`) from `models/LLM`, `models/text_encoders` and `models/clip`. Enable `use_lora` to pick a `.gguf` or `.safetensors` adapter from `models/loras` — GGUF nodes convert `.safetensors` adapters on first use, Transformers nodes apply them via PEFT. The single `MODEL_PATH` output carries a dict with the model path + format, the `use_lora` flag and the resolved LoRA path. Nothing is loaded into memory.  
- **QwenVL Prompt Enhancer (GGUF)**: GGUF text-only prompt enhancement.  

## **🧩 GGUF Nodes (llama.cpp backend)**

This repo includes **GGUF** nodes powered by `llama-cpp-python` (separate from the Transformers-based nodes).

- **Nodes**: `QwenVL (GGUF)`, `QwenVL (GGUF Advanced)`, `QwenVL Path Loader`, `QwenVL Prompt Enhancer (GGUF)`
- **Model folder** (default): `ComfyUI/models/llm/GGUF/` (configurable via `gguf_models.json`)
- **Custom model path**: connect `MODEL_PATH` from **QwenVL Path Loader** (or any string node) to the optional `model_path` input on the QwenVL GGUF nodes. The node then loads that file directly, and auto-detects `mmproj-model-bf16.gguf` (or any `*mmproj*.gguf`) in the same folder. This lets you use any custom Qwen VL GGUF stored in `models/LLM` or `models/text_encoders` without editing `gguf_models.json`.
- **LoRA**: enable `use_lora` in **QwenVL Path Loader**, pick an adapter from `models/loras`, then connect `MODEL_PATH` to the `model_path` input on the QwenVL GGUF node. The adapter is applied through llama.cpp's native LoRA support. Language-model LoRAs (attention/MLP, including Qwen3.5 linear-attention layers) are supported; vision-tower LoRAs are not. If no `adapter_config.json` is found next to the `.safetensors`, `lora_alpha` defaults to the adapter rank (scale 1.0).
- **Cancelling**: GGUF generation is streamed and checks ComfyUI's interrupt flag between tokens, so the **Cancel** button stops an active generation within about one token. Model loading and multimodal prompt prefill must finish first because llama.cpp does not expose an interruption point there.
- **Prompt Enhance FL2VA preset**: select `🎬 Prompt Enhance FL2VA` in `preset_prompt` and put the original idea in `custom_prompt`. The node loads the MiniMax-H3 prompt-writing guide (`AILab_FL2VA_Prompt_Guide.md`), uses the **first and last frames** of the connected `image` batch (or `video`) as `<Picture 1>` / `<Picture 2>`, and rewrites the idea into an `integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music` prompt. Set the clip length with the `duration (FL2VA-Ref2VA)` widget (default 8 s). With only one frame it falls back to I2VA rules; with no frame it falls back to T2VA. While this preset is active, `ctx` is raised to at least 16384 and `max_tokens` to at least 4096 so the guide and two reference frames fit.
- **Prompt Enhance Ref2VA preset**: select `🎬 Prompt Enhance Ref2VA` in `preset_prompt` and describe the target video in `custom_prompt`, referring to references as `<Picture N>` / `<Subject N>` (you may use `<Figure N>` as an alias). The node loads the MiniMax-H3 reference-conditioned guide (`AILab_Ref2VA_Prompt_Guide.md`) and uses **every frame of the `image` batch as an ordered reference picture** (`<Picture 1>`, `<Picture 2>`, …; up to 16). If no image is connected, frames sampled from the `video` input are used instead. The output follows the six-section Ref2VA layout: `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`. Reference labels in `custom_prompt` are localized to Chinese or English to match its language. `ctx` scales with the number of references (16384 base, +2048 per reference beyond two, capped at 32768) and `max_tokens` is raised to at least 4096. Only image references are supplied by this node; video/audio reference labels are not uploaded.
- **Vision requirement**: install a vision-capable `llama-cpp-python` wheel that provides `Qwen3VLChatHandler` / `Qwen25VLChatHandler`  
  See [docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md](docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md)

## **🗂️ Config Files**

- **HF models**: `hf_models.json`  
  - `hf_vl_models`: vision-language models (used by QwenVL nodes).  
  - `hf_text_models`: text-only models (used by Prompt Enhancer).  
- **GGUF models**: `gguf_models.json`  
- **System prompts**: `system_prompts.json` (includes both VL prompts and prompt-enhancer styles).  

## **📥 Download Models**

The models will be automatically downloaded on first use. If you prefer to download them manually, place them in the ComfyUI/models/LLM/Qwen-VL/ directory.

### **HF Vision Models (Qwen-VL)**
| Model | Link |
| :---- | :---- |
| Qwen3-VL-2B-Instruct | [Download](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
| Qwen3-VL-2B-Thinking | [Download](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking) |
| Qwen3-VL-2B-Instruct-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-FP8) |
| Qwen3-VL-2B-Thinking-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking-FP8) |
| Qwen3-VL-4B-Instruct | [Download](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| Qwen3-VL-4B-Thinking | [Download](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking) |
| Qwen3-VL-4B-Instruct-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8) |
| Qwen3-VL-4B-Thinking-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking-FP8) |
| Qwen3-VL-8B-Instruct | [Download](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Qwen3-VL-8B-Thinking | [Download](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) |
| Qwen3-VL-8B-Instruct-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8) |
| Qwen3-VL-8B-Thinking-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking-FP8) |
| Qwen3-VL-32B-Instruct | [Download](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| Qwen3-VL-32B-Thinking | [Download](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking) |
| Qwen3-VL-32B-Instruct-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8) |
| Qwen3-VL-32B-Thinking-FP8 | [Download](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking-FP8) |
| Qwen2.5-VL-3B-Instruct | [Download](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| Qwen2.5-VL-7B-Instruct | [Download](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |

### **HF Text Models (Qwen3)**
| Model | Link |
| :---- | :---- |
| Qwen3-0.6B | [Download](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Qwen3-4B-Instruct-2507 | [Download](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| qwen3-4b-Z-Image-Engineer | [Download](https://huggingface.co/BennyDaBall/qwen3-4b-Z-Image-Engineer) |

### **GGUF Models (Manual Download)**
| Group | Model | Repo | Alt Repo | Model Files | MMProj |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Qwen text (GGUF) | Qwen3-4B-GGUF | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) |  | Qwen3-4B-Q4_K_M.gguf, Qwen3-4B-Q5_0.gguf, Qwen3-4B-Q5_K_M.gguf, Qwen3-4B-Q6_K.gguf, Qwen3-4B-Q8_0.gguf |  |
| Qwen-VL (GGUF) | Qwen3-VL-4B-Instruct-GGUF | [Qwen/Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) |  | Qwen3VL-4B-Instruct-F16.gguf, Qwen3VL-4B-Instruct-Q4_K_M.gguf, Qwen3VL-4B-Instruct-Q8_0.gguf | mmproj-Qwen3VL-4B-Instruct-F16.gguf |
| Qwen-VL (GGUF) | Qwen3-VL-8B-Instruct-GGUF | [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) |  | Qwen3VL-8B-Instruct-F16.gguf, Qwen3VL-8B-Instruct-Q4_K_M.gguf, Qwen3VL-8B-Instruct-Q8_0.gguf | mmproj-Qwen3VL-8B-Instruct-F16.gguf |
| Qwen-VL (GGUF) | Qwen3-VL-4B-Thinking-GGUF | [Qwen/Qwen3-VL-4B-Thinking-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking-GGUF) |  | Qwen3VL-4B-Thinking-F16.gguf, Qwen3VL-4B-Thinking-Q4_K_M.gguf, Qwen3VL-4B-Thinking-Q8_0.gguf | mmproj-Qwen3VL-4B-Thinking-F16.gguf |
| Qwen-VL (GGUF) | Qwen3-VL-8B-Thinking-GGUF | [Qwen/Qwen3-VL-8B-Thinking-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking-GGUF) |  | Qwen3VL-8B-Thinking-F16.gguf, Qwen3VL-8B-Thinking-Q4_K_M.gguf, Qwen3VL-8B-Thinking-Q8_0.gguf | mmproj-Qwen3VL-8B-Thinking-F16.gguf |
| Qwen-VL (GGUF) | Qwen3.5-VL-7B-Instruct-GGUF | [Qwen/Qwen3.5-VL-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3.5-VL-7B-Instruct-GGUF) |  | Qwen3.5VL-7B-Instruct-F16.gguf, Qwen3.5VL-7B-Instruct-Q4_K_M.gguf, Qwen3.5VL-7B-Instruct-Q8_0.gguf | mmproj-Qwen3.5VL-7B-Instruct-F16.gguf |
| Qwen-VL (GGUF) | Qwen3.6-VL-MoE-Instruct-GGUF | [Qwen/Qwen3.6-VL-MoE-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3.6-VL-MoE-Instruct-GGUF) |  | Qwen3.6VL-MoE-Instruct-F16.gguf, Qwen3.6VL-MoE-Instruct-Q4_K_M.gguf, Qwen3.6VL-MoE-Instruct-Q8_0.gguf | mmproj-Qwen3.6VL-MoE-Instruct-F16.gguf |
| Qwen-VL (GGUF) | Qwen3.8-VL-14B-Instruct-GGUF | [Qwen/Qwen3.8-VL-14B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3.8-VL-14B-Instruct-GGUF) |  | Qwen3.8VL-14B-Instruct-F16.gguf, Qwen3.8VL-14B-Instruct-Q4_K_M.gguf, Qwen3.8VL-14B-Instruct-Q8_0.gguf | mmproj-Qwen3.8VL-14B-Instruct-F16.gguf |

## **📖 Usage**

### **Basic Usage**

1. Add the **"QwenVL"** node from the 🧪AILab/QwenVL category.  
2. Select the **model\_name** you wish to use.  
3. Connect an image or video (image sequence) source to the node.  
4. Write your prompt using the preset or custom field.  
5. Run the workflow.

### **Advanced Usage**

For more control, use the **"QwenVL (Advanced)"** node. This gives you access to detailed generation parameters like temperature, top\_p, beam search, and device selection.

## **⚙️ Parameters**

| Parameter | Description | Default | Range | Node(s) |
| :---- | :---- | :---- | :---- | :---- |
| **model\_name** | The Qwen-VL model to use. | Qwen3-VL-4B-Instruct | \- | Standard & Advanced |
| **quantization** | On-the-fly quantization. Ignored for pre-quantized models (e.g., FP8). | 8-bit (Balanced) | 4-bit, 8-bit, None | Standard & Advanced |
| **attention\_mode** | Attention mechanism: auto (Sage→Flash→SDPA), sage, flash\_attention\_2, sdpa | auto | auto, sage, flash\_attention\_2, sdpa | Standard & Advanced |
| **preset\_prompt** | A selection of pre-defined prompts for common tasks. | "Describe this..." | Any text | Standard & Advanced |
| **custom\_prompt** | Overrides the preset prompt if provided. |  | Any text | Standard & Advanced |
| **max\_tokens** | Maximum number of new tokens to generate. | 1024 | 64-2048 | Standard & Advanced |
| **keep\_model\_loaded** | Keep the model in VRAM for faster subsequent runs. | True | True/False | Standard & Advanced |
| **seed** | A seed for reproducible results. | 1 | 1 \- 2^64-1 | Standard & Advanced |
| **temperature** | Controls randomness. Higher values \= more creative. (Used when num\_beams is 1). | 0.6 | 0.1-1.0 | Advanced Only |
| **top\_p** | Nucleus sampling threshold. (Used when num\_beams is 1). | 0.9 | 0.0-1.0 | Advanced Only |
| **num\_beams** | Number of beams for beam search. \> 1 disables temperature/top\_p sampling. | 1 | 1-10 | Advanced Only |
| **repetition\_penalty** | Discourages repeating tokens. | 1.2 | 0.0-2.0 | Advanced Only |
| **frame\_count** | Number of frames to sample from the video input. | 16 | 1-64 | Advanced Only |
| **device** | Override automatic device selection. | auto | auto, cuda, cpu | Advanced Only |
| **use\_torch\_compile** | Enable torch.compile optimization for faster inference. | False | True/False | Advanced Only |

### **💡 Quantization Options**

| Mode | Precision | Memory Usage | Speed | Quality | Recommended For |
| :---- | :---- | :---- | :---- | :---- | :---- |
| None (FP16) | 16-bit Float | High | Fastest | Best | High VRAM GPUs (16GB+) |
| 8-bit (Balanced) | 8-bit Integer | Medium | Fast | Very Good | Balanced performance (8GB+) |
| 4-bit (VRAM-friendly) | 4-bit Integer | Low | Slower\* | Good | Low VRAM GPUs (<8GB) |

\* **Note on 4-bit Speed**: 4-bit quantization significantly reduces VRAM usage but may result in slower performance on some systems due to the computational overhead of real-time dequantization.

### **🎯 Attention Mode Guide**

| Mode | Description | Best For |
| :---- | :---- | :---- |
| **auto** | Automatically selects best available: Sage → Flash → SDPA | Most users (recommended) |
| **sage** | SageAttention with GPU-optimized kernels | Speed on modern GPUs (RTX 40 series, Hopper, Blackwell) |
| **flash\_attention\_2** | Flash Attention 2 | Speed when Sage unavailable |
| **sdpa** | PyTorch SDPA (default) | Compatibility, FP8/BitsAndBytes models |

**Note**: FP8 models and BitsAndBytes quantization automatically use SDPA regardless of selection.

### **🤔 Setting Tips**

| Setting | Recommendation |
| :---- | :---- |
| **Model Choice** | For most users, Qwen3-VL-4B-Instruct is a great starting point. If you have a 40-series GPU, try the \-FP8 version for better performance. |
| **Memory Mode** | Keep keep\_model\_loaded enabled (True) for the best performance if you plan to run the node multiple times. Disable it only if you are running out of VRAM for other nodes. |
| **Quantization** | Start with the default 8-bit. If you have plenty of VRAM (>16GB), switch to None (FP16) for the best speed and quality. If you are low on VRAM, use 4-bit. |
| **Attention Mode** | Use "auto" for best performance. SageAttention provides fastest inference on supported GPUs. |
| **Performance** | The first time a model is loaded with a specific quantization, it may be slow. Subsequent runs (with keep\_model\_loaded enabled) will be much faster. |

## **🙏 Credits**

* **Qwen Team**: [Alibaba Cloud](https://github.com/QwenLM) \- For developing and open-sourcing the powerful Qwen-VL models.  
* **ComfyUI**: [comfyanonymous](https://github.com/comfyanonymous/ComfyUI) \- For the incredible and extensible ComfyUI platform.  
* **llama-cpp-python**: [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) \- GGUF backend with vision support used by the GGUF nodes.  
* **SageAttention**: [SageAttention](https://github.com/thu-ml/SageAttention) \- Efficient attention implementation with GPU-optimized kernels.
* **ComfyUI Integration**: [1038lab](https://github.com/1038lab) \- Developer of this custom node.

## **📜 License**

This repository's code is released under the [GPL-3.0 License](LICENSE).
