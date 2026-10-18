from pathlib import Path

report_path = Path("reports/final_project_report.md")
content = report_path.read_text(encoding="utf-8")

old_sec7 = """### 7.2 Detailed Layer-by-Layer Parameter Inventory

| Layer Index | Layer Type | Kernel Size | Stride | Input Channels | Output Channels | Trainable Parameters | Activation Output Shape |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Input Spectrogram | — | — | — | 1 | 0 | `[1, 40, 47, 1]` |
| 1 | Standard Conv2D | $3 \\times 3$ | 2 | 1 | 28 | 280 (252 w + 28 b) | `[1, 20, 24, 28]` |
| 2 | BatchNorm + ReLU | — | — | 28 | 28 | 56 (folded in TFLite)| `[1, 20, 24, 28]` |
| 3 | Depthwise Conv2D 1 | $3 \\times 3$ | 1 | 28 | 28 | 280 (252 w + 28 b) | `[1, 20, 24, 28]` |
| 4 | BatchNorm + ReLU | — | — | 28 | 28 | 56 (folded in TFLite)| `[1, 20, 24, 28]` |
| 5 | Pointwise Conv2D 1 | $1 \\times 1$ | 1 | 28 | 28 | 812 (784 w + 28 b) | `[1, 20, 24, 28]` |
| 6 | BatchNorm + ReLU | — | — | 28 | 28 | 56 (folded in TFLite)| `[1, 20, 24, 28]` |
| 7 | Depthwise Conv2D 2 | $3 \\times 3$ | 1 | 28 | 28 | 280 (252 w + 28 b) | `[1, 20, 24, 28]` |
| 8 | BatchNorm + ReLU | — | — | 28 | 28 | 56 (folded in TFLite)| `[1, 20, 24, 28]` |
| 9 | Pointwise Conv2D 2 | $1 \\times 1$ | 1 | 28 | 28 | 812 (784 w + 28 b) | `[1, 20, 24, 28]` |
| 10| BatchNorm + ReLU | — | — | 28 | 28 | 56 (folded in TFLite)| `[1, 20, 24, 28]` |
| 11| AdaptiveAvgPool2D | — | — | 28 | 28 | 0 | `[1, 1, 1, 28]` |
| 12| Flatten + Linear | — | — | 28 | 12 | 348 (336 w + 12 b) | `[1, 12]` |
| — | **Total Architecture**| — | — | — | — | **5,248 Parameters** | — |"""

new_sec7 = """### 7.2 Detailed Layer-by-Layer Parameter Inventory (4 DS Blocks)

The actual DS-CNN implementation (`src/models/dscnn.py`) comprises **4 Depthwise Separable Blocks** (`ds_blocks = 4`). All convolutions use `bias=False` followed by Batch Normalization (which provides trainable scale $\\gamma$ and shift $\\beta$ parameters):

| Layer Index | Layer Type | Kernel Size | Stride | Input Channels | Output Channels | Trainable Parameters | Activation Output Shape |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Input Spectrogram | — | — | — | 1 | 0 | `[1, 40, 47, 1]` |
| 1 | Stem Conv2D (bias=False) | $3 \\times 3$ | 2 | 1 | 28 | 252 weights | `[1, 20, 24, 28]` |
| 2 | Stem BatchNorm + ReLU | — | — | 28 | 28 | 56 (28 w + 28 b) | `[1, 20, 24, 28]` |
| 3 | DS Block 0: Depthwise Conv | $3 \\times 3$ | 1 | 28 | 28 | 252 weights | `[1, 20, 24, 28]` |
| 4 | DS Block 0: BN + ReLU | — | — | 28 | 28 | 56 (28 w + 28 b) | `[1, 20, 24, 28]` |
| 5 | DS Block 0: Pointwise Conv | $1 \\times 1$ | 1 | 28 | 28 | 784 weights | `[1, 20, 24, 28]` |
| 6 | DS Block 0: BN + ReLU | — | — | 28 | 28 | 56 (28 w + 28 b) | `[1, 20, 24, 28]` |
| 7 | DS Block 1: Depthwise Conv | $3 \\times 3$ | 1 | 28 | 28 | 252 weights | `[1, 20, 24, 28]` |
| 8 | DS Block 1: BN + ReLU | — | — | 28 | 28 | 56 (28 w + 28 b) | `[1, 20, 24, 28]` |
| 9 | DS Block 1: Pointwise Conv | $1 \\times 1$ | 1 | 28 | 28 | 784 weights | `[1, 20, 24, 28]` |
| 10| DS Block 1: BN + ReLU | — | — | 28 | 28 | 56 (28 w + 28 b) | `[1, 20, 24, 28]` |
| 11| DS Block 2: Depthwise Conv | $3 \\times 3$ | 1 | 28 | 28 | 252 weights | `[1, 20, 24, 28]` |
| 12| DS Block 2: BN + ReLU | — | — | 28 | 28 | 56 (28 w + 28 b) | `[1, 20, 24, 28]` |
| 13| DS Block 2: Pointwise Conv | $1 \\times 1$ | 1 | 28 | 28 | 784 weights | `[1, 20, 24, 28]` |
| 14| DS Block 2: BN + ReLU | — | — | 28 | 28 | 56 (28 w + 28 b) | `[1, 20, 24, 28]` |
| 15| DS Block 3: Depthwise Conv | $3 \\times 3$ | 1 | 28 | 28 | 252 weights | `[1, 20, 24, 28]` |
| 16| DS Block 3: BN + ReLU | — | — | 28 | 28 | 56 (28 w + 28 b) | `[1, 20, 24, 28]` |
| 17| DS Block 3: Pointwise Conv | $1 \\times 1$ | 1 | 28 | 28 | 784 weights | `[1, 20, 24, 28]` |
| 18| DS Block 3: BN + ReLU | — | — | 28 | 28 | 56 (28 w + 28 b) | `[1, 20, 24, 28]` |
| 19| AdaptiveAvgPool2D | — | — | 28 | 28 | 0 | `[1, 1, 1, 28]` |
| 20| Flatten + Linear Classifier | — | — | 28 | 12 | 348 (336 w + 12 b) | `[1, 12]` |
| — | **Total Architecture**| — | — | — | — | **5,248 Parameters** | — |

### 7.3 Computational Complexity (MACs & FLOPs)
* **Stem Conv2D:** $20 \\times 24 \\times (1 \\times 3 \\times 3) \\times 28 = 120,960\\text{ MACs}$
* **Single DS Block:** $(20 \\times 24 \\times 9 \\times 28) + (20 \\times 24 \\times 28 \\times 28) = 120,960 + 376,320 = 497,280\\text{ MACs}$
* **4 DS Blocks:** $4 \\times 497,280 = 1,989,120\\text{ MACs}$
* **Linear Classifier:** $28 \\times 12 = 336\\text{ MACs}$
* **Total Multiply-Accumulate Operations (MACs):** **2,110,416 MACs (2.110 MMACs)**
* **Total Floating-Point Operations (FLOPs, 2× MACs):** **4,220,832 FLOPs (4.221 MFLOPs)**"""

assert old_sec7 in content, "old_sec7 not found in content"
content = content.replace(old_sec7, new_sec7)

# Also update section 12.2 safety margin description
old_margin = "* **Engineering Safety Margin:** `30,720 B - 28,928 B = 1,792 bytes` (**+10.7% margin**), ensuring zero allocation failure during model initialization."
new_margin = "* **Engineering Safety Margin:** `30,720 B - 28,928 B = 1,792 bytes` (**+5.8% of configured arena / +6.2% above analytical minimum**), ensuring zero allocation failure during model initialization."
assert old_margin in content, "old_margin not found in content"
content = content.replace(old_margin, new_margin)

report_path.write_text(content, encoding="utf-8")
print("Updated final_project_report.md successfully!")
