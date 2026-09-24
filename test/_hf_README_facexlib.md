---
license: mit
tags:
  - face-detection
  - face-alignment
  - face-parsing
  - face-restoration
  - computer-vision
  - python-3-14
  - wheels
  - comfyui
  - stable-diffusion
  - forge
library_name: facexlib
---

# FaceXLib (Universal Python 3.12 / 3.13 / 3.14 Compatible Build)

Universal, pure-Python wheel distribution for **`facexlib`** (v0.3.0), optimized for modern Python runtime environments (Python 3.10 through Python 3.14+).

Original upstream repository: [xinntao/facexlib](https://github.com/xinntao/facexlib)

---

## Overview

`facexlib` is a foundational computer vision library providing standardized face processing modules (detection, alignment, parsing, tracking, assessment, and restoration preprocessing) widely utilized across AI generation and restoration ecosystems, including **ComfyUI**, **Stable Diffusion WebUI Forge**, **GFPGAN**, **CodeFormer**, **RestoreFormer**, **IOPaint**, and **ReActor**.

Standard upstream releases pin legacy dependencies (specifically `numba` and restrictive `filterpy` builds) that frequently fail to compile or install on modern Python releases (Python 3.12, 3.13, and 3.14). This universal build solves those environment blockers while preserving 100% API and functional compatibility.

---

## Key Improvements in this Release

1. **Numba-Free / Safe JIT Fallback**:
   - Upstream `data_association.py` hard-required `numba.jit`, causing installation and import failures in Python 3.12+ environments where prebuilt Numba wheels were unavailable.
   - Replaced with a graceful fallback wrapper (`try ... except ImportError`) that defaults to native vectorized NumPy/SciPy operations when Numba is not installed.
2. **Cleaned & Minimal Dependency Tree**:
   - Removed strict dependency pins. Core requirements are lightweight and modern:
     - `numpy`
     - `opencv-python`
     - `Pillow`
     - `scipy`
     - `tqdm`
3. **Universal Pure-Python Wheel (`py3-none-any.whl`)**:
   - Platform-independent (Windows, Linux, macOS) and architecture-independent (x86_64, ARM64/Apple Silicon).
   - Zero C/C++ compilation requirements during `pip install`.

---

## Installation

### Direct Install via pip

```bash
pip install https://huggingface.co/ussoewwin/facexlib/resolve/main/facexlib-0.3.0-py3-none-any.whl
```

### In `requirements.txt`

```text
facexlib @ https://huggingface.co/ussoewwin/facexlib/resolve/main/facexlib-0.3.0-py3-none-any.whl
```

---

## Core Capabilities

| Module | Available Backends / Models | Typical Use Case |
| :--- | :--- | :--- |
| **`detection`** | RetinaFace (`resnet50`, `mobile0.25`), YOLOv5-face | High-precision face bounding box & 5-point landmark detection |
| **`alignment`** | 5-point similarity transformation, cropped affine warping | Face normalization for restoration models (GFPGAN / CodeFormer) |
| **`parsing`** | BiSeNet (19-class semantic segmentation) | Hair, skin, eye, mouth, and accessory segmentation |
| **`tracking`** | SORT with Kalman Filter | Real-time temporal face association in video pipelines |
| **`assessment`** | HyperIQA, MUSIQ | No-reference image and facial perceptual quality evaluation |
| **`recognition`** | ArcFace | Deep facial feature extraction and verification |

---

## Quick Start Example

### 1. Face Detection & Landmark Extraction

```python
import cv2
import torch
from facexlib.detection import init_detection_model, detect_faces

# Initialize RetinaFace detector (auto-downloads weights on first run)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
det_net = init_detection_model("retinaface_resnet50", half=False, device=device)

# Load image (BGR)
img = cv2.imread("input.jpg")

# Detect faces
with torch.no_grad():
    bboxes = detect_faces(det_net, img, device=device)

print(f"Detected {len(bboxes)} faces.")
# bboxes format: [[x1, y1, x2, y2, score, landmark_5x2...], ...]
```

### 2. Face Semantic Parsing (BiSeNet)

```python
import torch
from facexlib.parsing import init_parsing_model, parsenet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parse_net = init_parsing_model(model_name="bisenet", device=device)

# Input: Cropped & aligned 512x512 face tensor
with torch.no_grad():
    # out tensor contains 19-class segmentation logits
    pass
```

---

## Automatic Weight Management

Pretrained model weights continue to be automatically fetched and cached in standard local directories on demand:
- Windows: `%USERPROFILE%/.cache/facexlib/weights/`
- Linux / macOS: `~/.cache/facexlib/weights/`

---

## License

- **Library & Code**: MIT License.
- **Underlying Model Checkpoints**: Subject to their respective original licenses (RetinaFace, BiSeNet, ArcFace, etc.).
