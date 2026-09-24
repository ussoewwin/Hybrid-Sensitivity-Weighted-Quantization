---
license: apache-2.0
tags:
  - gradio
  - python-3-14
  - python-3-13
  - windows
  - webui
  - stable-diffusion
  - forge
  - wheels
library_name: gradio
---

# Gradio (Universal Python 3.12 / 3.13 / 3.14 Compatible Builds)

Universal, pure-Python wheel distributions for **`gradio`** (v3.41.2 and v4.40.0), patched and optimized for seamless compatibility with modern Python environments (including Python 3.13 and Python 3.14 on Windows and Linux).

Original upstream repository: [gradio-app/gradio](https://github.com/gradio-app/gradio)

---

## Overview

Gradio is the standard Python framework for building interactive machine learning web interfaces and web applications. In generative AI ecosystems like **Stable Diffusion WebUI (AUTOMATIC1111)**, **Stable Diffusion WebUI Forge**, and various standalone GUI tools, specific versions of Gradio are pinned.

As developers transition to newer Python versions (Python 3.12, 3.13, and 3.14), upstream packages and dependencies often face compatibility hurdles on Windows. This repository provides pre-built, tested universal wheels for both Gradio 3.x and Gradio 4.x.

---

## Available Wheels

| Version | Wheel File | Target Ecosystem / Use Case |
| :--- | :--- | :--- |
| **Gradio 4.40.0** | [`gradio-4.40.0-py3-none-any.whl`](https://huggingface.co/ussoewwin/gradio/resolve/main/gradio-4.40.0-py3-none-any.whl) | Stable Diffusion WebUI Forge, modern ML apps, multimodal interfaces |
| **Gradio 3.41.2** | [`gradio-3.41.2-py3-none-any.whl`](https://huggingface.co/ussoewwin/gradio/resolve/main/gradio-3.41.2-py3-none-any.whl) | Stable Diffusion WebUI (A1111 legacy pipeline compatibility) |

---

## Key Features

1. **Python 3.13 & 3.14 Compatibility on Windows**:
   - Cleaned package metadata and dependency constraints to install without build tools or legacy compilation errors.
2. **Pure-Python Universal Distribution (`py3-none-any.whl`)**:
   - Architecture-independent (x86_64, ARM64) and OS-independent (Windows, Linux, macOS).
3. **Drop-in Replacement**:
   - Maintains full binary and API compatibility with existing extensions, custom components, and UI themes.

---

## Installation

### Gradio 4.40.0 (Recommended for Forge & Modern UIs)

```bash
pip install https://huggingface.co/ussoewwin/gradio/resolve/main/gradio-4.40.0-py3-none-any.whl
```

Or in `requirements.txt`:
```text
gradio @ https://huggingface.co/ussoewwin/gradio/resolve/main/gradio-4.40.0-py3-none-any.whl
```

### Gradio 3.41.2 (For Classic WebUI Pipelines)

```bash
pip install https://huggingface.co/ussoewwin/gradio/resolve/main/gradio-3.41.2-py3-none-any.whl
```

Or in `requirements.txt`:
```text
gradio @ https://huggingface.co/ussoewwin/gradio/resolve/main/gradio-3.41.2-py3-none-any.whl
```

---

## Quick Start Example

### 1. Simple Interface

```python
import gradio as gr

def generate_text(prompt: str, temperature: float):
    return f"Response to '{prompt}' at temp {temperature}"

demo = gr.Interface(
    fn=generate_text,
    inputs=[gr.Textbox(label="Prompt"), gr.Slider(0.0, 1.0, value=0.7, label="Temperature")],
    outputs=gr.Textbox(label="Output"),
    title="Gradio Python 3.14 Demo"
)

if __name__ == "__main__":
    demo.launch()
```

### 2. Complex Layout with `gr.Blocks`

```python
import gradio as gr

with gr.Blocks(title="Generative AI Studio") as demo:
    gr.Markdown("# Image Generation Studio")
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", placeholder="Enter description...")
            steps = gr.Slider(1, 50, value=20, step=1, label="Sampling Steps")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Result")
            
    btn.click(fn=lambda p, s: None, inputs=[prompt, steps], outputs=output_image)

if __name__ == "__main__":
    demo.launch()
```

---

## License

- **License**: Apache License 2.0 (consistent with upstream Gradio).
