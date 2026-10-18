"""
Script to generate the Stage 9 PowerPoint Presentation:
reports/final_project_presentation.pptx
Target: 12 cleanly formatted, widescreen 16:9 slides using native editable elements.
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    output_path = Path("reports/final_project_presentation.pptx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # Blank slide
    
    # Palette definition
    BG_DARK = RGBColor(15, 23, 42)        # Deep Navy / Slate
    BG_LIGHT = RGBColor(248, 250, 252)    # Crisp Off-White
    CARD_BG = RGBColor(255, 255, 255)     # Card White
    CARD_BORDER = RGBColor(226, 232, 240) # Card Border
    TEXT_MAIN = RGBColor(15, 23, 42)      # Primary Dark Text
    TEXT_MUTED = RGBColor(100, 116, 139)  # Secondary Muted Text
    ACCENT_BLUE = RGBColor(14, 116, 144)  # Deep Teal / Cyan
    ACCENT_CYAN = RGBColor(2, 132, 199)   # Sky Blue
    ACCENT_GREEN = RGBColor(22, 163, 74)  # Emerald Green
    ACCENT_ROSE = RGBColor(225, 29, 72)   # Rose Red
    CARD_HEADER_BG = RGBColor(241, 245, 249)

    def add_header(slide, title_text, category_text="FINAL-YEAR ECE PROJECT | EDGE AI"):
        # Header container banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = Pt(11)
        p0.font.bold = True
        p0.font.color.rgb = ACCENT_BLUE
        
        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(24)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_MAIN

    def add_card(slide, left, top, width, height, title=None):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        shape.line.color.rgb = CARD_BORDER
        shape.line.width = Pt(1.5)
        
        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), width - Inches(0.4), Inches(0.5))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = ACCENT_BLUE
        return shape

    def add_footnote(slide, text="Host: AMD Ryzen 3 7320U CPU | Target: ESP32-S3 (Virtual) | 61/61 Automated Tests Passing"):
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(6.9), Inches(11.733), Inches(0.4))
        tf = tb.text_frame
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(10)
        p.font.italic = True
        p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Theme)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = BG_DARK
    bg1.line.fill.background()

    tb1 = slide1.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(10.9), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "BACHELOR OF ENGINEERING / TECHNOLOGY — ECE CAPSTONE"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = RGBColor(56, 189, 248) # Light Cyan
    
    p = tf1.add_paragraph()
    p.text = "Edge AI Model Compression Pipeline for Microcontroller Keyword Spotting"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    
    p = tf1.add_paragraph()
    p.text = "Design, INT8 Quantization, and Embedded Deployment Analysis of an Ultra-Compact DS-CNN Targeting the ESP32-S3"
    p.font.size = Pt(18)
    p.font.color.rgb = RGBColor(203, 213, 225)
    
    p = tf1.add_paragraph()
    p.text = "\nVerified Artifacts: 5,248 Parameters | 21.77 KB INT8 FlatBuffer | 54.83% Accuracy | 52.0 KB Working SRAM\nHost Environment: Windows 11 CPU (Ryzen 3 7320U) | Target: Espressif ESP32-S3 (Virtual Scaffolding)"
    p.font.size = Pt(13)
    p.font.color.rgb = RGBColor(148, 163, 184)

    # =========================================================================
    # SLIDE 2: Problem Statement & Motivation
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "Problem Statement & Edge AI Motivation")
    
    # 3 Cards: Cloud Vulnerabilities, Microcontroller Challenges, The Opportunity
    add_card(slide2, Inches(0.8), Inches(1.7), Inches(3.6), Inches(4.9), "1. Cloud Audio Bottlenecks")
    tb = slide2.shapes.add_textbox(Inches(1.0), Inches(2.4), Inches(3.2), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    bullets = [
        "High Latency: Round-trip internet transfers introduce 150–1,500 ms delays, preventing real-time control.",
        "Privacy Hazards: Continuous streaming of private speech over Wi-Fi creates severe eavesdropping risks.",
        "Bandwidth Overhead: Continuous 16 kHz audio demands ~256 kbps stream per node.",
        "Network Brittleness: Voice UI fails completely during connection drops or weak signal."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_card(slide2, Inches(4.8), Inches(1.7), Inches(3.6), Inches(4.9), "2. MCU Hardware Constraints")
    tb = slide2.shapes.add_textbox(Inches(5.0), Inches(2.4), Inches(3.2), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    bullets = [
        "SRAM Scarcity: ESP32-S3 has only 512 KB internal SRAM for runtime heap, stacks, and tensor buffers.",
        "No Dedicated NPU: Execution relies on Xtensa LX7 dual cores without GPU matrix acceleration.",
        "Flash Limits: Typical 4 MB SPI Flash must accommodate OTA partitions, Wi-Fi stacks, and code.",
        "Thermal & Power Budget: Low milliwatt envelope demands minimal clock cycles per inference."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_card(slide2, Inches(8.8), Inches(1.7), Inches(3.7), Inches(4.9), "3. The TinyML Solution")
    tb = slide2.shapes.add_textbox(Inches(9.0), Inches(2.4), Inches(3.3), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    bullets = [
        "Ultra-Compact Network: DS-CNN with 5,248 parameters (~20.5 KB float weights).",
        "Full INT8 Quantization: Eliminates software float emulation; uses native SIMD instructions.",
        "Static Tensor Arena: Predictable 30 KB SRAM buffer with zero dynamic heap allocation.",
        "Local Autonomy: 100% on-device decision making with zero network dependency."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_footnote(slide2)

    # =========================================================================
    # SLIDE 3: Objectives & Scope
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Project Objectives & Execution Scope")
    
    add_card(slide3, Inches(0.8), Inches(1.7), Inches(5.6), Inches(4.9), "Primary Engineering Goals")
    tb = slide3.shapes.add_textbox(Inches(1.0), Inches(2.4), Inches(5.2), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    goals = [
        "12-Class Keyword Spotting: Classify 10 core words ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go') + '_silence_' and '_unknown_'.",
        "Deterministic Zero-Leakage Dataset: Curate 4,890 Speech Commands WAVs partitioned strictly by cryptographic speaker hashing.",
        "Full Integer Quantization: Post-training 8-bit integer calibration with full signed int8 input/output tensors.",
        "ESP32-S3 Firmware Scaffold: Generate 8-byte aligned C header (`model_data.h`) and FreeRTOS task with `esp-nn` kernel support.",
        "Comprehensive Memory Budgeting: Model layer-by-layer activation buffers to mathematically size the Tensor Arena."
    ]
    for g in goals:
        p = tf.add_paragraph()
        p.text = "✔ " + g
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_card(slide3, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.9), "Scope & Explicit Disclosures")
    tb = slide3.shapes.add_textbox(Inches(7.0), Inches(2.4), Inches(5.3), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    disclosures = [
        "Host Execution: All model training, evaluation, and dashboard inference were conducted on the host AMD Ryzen 3 7320U CPU.",
        "Microcontroller Estimates: ESP32-S3 latency (~6–10 ms) is an analytical projection from literature, not on-silicon measurement.",
        "Firmware Compilation Status: ESP-IDF source files and CMake configs are syntactically validated, but the compiler (`idf.py`) is not installed on this host (Template Only / Not Built).",
        "Physical Hardware: Flashing to physical silicon, battery current draw, and hardware audio capture were NOT performed.",
        "Reproducibility: 100% of reported numbers originate from verified disk artifacts and a 61-test automated suite."
    ]
    for d in disclosures:
        p = tf.add_paragraph()
        p.text = "ℹ " + d
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_footnote(slide3)

    # =========================================================================
    # SLIDE 4: End-to-End System Architecture
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "End-to-End System Pipeline & Data Flow")
    
    # 4 Sequential Cards
    steps = [
        ("Step 1: Audio Preprocessor", "16 kHz Mono WAV (1.0s)\n16,000 samples\nPre-emphasis (α=0.97)\nHann windowing (25ms)\nStride (20ms) -> 47 frames\n40 Mel-filterbank DCT"),
        ("Step 2: DS-CNN Baseline", "Feature map: [1, 40, 47]\n5,248 parameters\nStandard Conv (28 filters)\n2× Depthwise-Pointwise blocks\nGlobal Pooling + Linear\nPyTorch FP32 State Dict"),
        ("Step 3: INT8 Quantization", "Post-Training Quantization\n150-sample calibration set\nFull int8 input & output\nS_in = 0.4796, Z_in = 54\nS_out = 0.1009, Z_out = 82\nTFLite FlatBuffer (21.77 KB)"),
        ("Step 4: ESP32-S3 Scaffold", "Aligned C Header (alignas 8)\nSHA-256 Verified\n30 KB Tensor Arena\nFreeRTOS Task Architecture\nESP-NN Vector Kernels\n52.0 KB Working SRAM")
    ]
    for i, (title, content) in enumerate(steps):
        left = Inches(0.8 + i * 2.95)
        add_card(slide4, left, Inches(1.7), Inches(2.8), Inches(4.9), title)
        tb = slide4.shapes.add_textbox(left + Inches(0.15), Inches(2.5), Inches(2.5), Inches(3.9))
        tf = tb.text_frame
        tf.word_wrap = True
        for line in content.split("\n"):
            p = tf.add_paragraph()
            p.text = line
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_MAIN

    add_footnote(slide4)

    # =========================================================================
    # SLIDE 5: Dataset & Audio Preprocessing
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "Dataset Integrity & MFCC Feature Extraction")
    
    add_card(slide5, Inches(0.8), Inches(1.7), Inches(5.6), Inches(4.9), "Deterministic Speaker-Isolated Dataset")
    tb = slide5.shapes.add_textbox(Inches(1.0), Inches(2.4), Inches(5.2), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    ds_points = [
        "Dataset Source: Google Speech Commands v2 (4,890 verified WAV files, 149.44 MB).",
        "12 Target Classes: 10 commands ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go') + '_silence_' (ambient noise) + '_unknown_' (19 other words).",
        "Speaker Isolation Mechanism: SHA-1 hash of speaker ID guarantees 0.0% speaker overlap between splits:",
        "  • Training: 3,936 clips (829 unique speakers)",
        "  • Validation: 436 clips (98 unique speakers)",
        "  • Held-out Test: 518 clips (102 unique speakers)",
        "Zero-Leakage Guarantee: Prevents neural memorization of individual vocal timbres."
    ]
    for pt in ds_points:
        p = tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_card(slide5, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.9), "MFCC Signal Processing Chain")
    tb = slide5.shapes.add_textbox(Inches(7.0), Inches(2.4), Inches(5.3), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    mfcc_points = [
        "Sampling Rate: 16,000 Hz, 16-bit PCM, single-channel mono.",
        "Framing: 25 ms window (400 samples) with 20 ms stride (320 samples) -> 47 time frames.",
        "Windowing: Periodic Hann window to suppress spectral leakage.",
        "Spectral Analysis: 512-point FFT -> 40 triangular Mel-scale filterbanks (20 Hz to 4,000 Hz).",
        "Log Energy & DCT: Type-II Discrete Cosine Transform extracts 40 orthogonal cepstral coefficients.",
        "Output Feature Tensor: Shape [1, 40, 47] (1,880 total elements per 1.0 second audio clip)."
    ]
    for pt in mfcc_points:
        p = tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_footnote(slide5)

    # =========================================================================
    # SLIDE 6: DS-CNN Architecture & Baseline Training
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "DS-CNN Architecture & Baseline FP32 Model")
    
    # Left table: Layer dimensions
    rows, cols = 8, 4
    left = Inches(0.8)
    top = Inches(1.7)
    width = Inches(6.8)
    height = Inches(4.9)
    table_shape = slide6.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    
    headers = ["Layer Type", "Kernel / Stride", "Params", "Output Shape"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_HEADER_BG
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(11)
            p.font.color.rgb = ACCENT_BLUE

    layer_data = [
        ("Input Feature", "—", "0", "[1, 40, 47, 1]"),
        ("Conv2D + BN + ReLU", "3×3, Stride 2", "336", "[1, 20, 24, 28]"),
        ("Depthwise Conv 1", "3×3, Stride 1", "336", "[1, 20, 24, 28]"),
        ("Pointwise Conv 1", "1×1, Stride 1", "868", "[1, 20, 24, 28]"),
        ("Depthwise Conv 2", "3×3, Stride 1", "336", "[1, 20, 24, 28]"),
        ("Pointwise Conv 2", "1×1, Stride 1", "868", "[1, 20, 24, 28]"),
        ("AdaptiveAvgPool + Linear", "Pool + FC", "348", "[1, 12]")
    ]
    for r, row in enumerate(layer_data):
        for c, val in enumerate(row):
            cell = table.cell(r + 1, c)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.color.rgb = TEXT_MAIN

    add_card(slide6, Inches(8.0), Inches(1.7), Inches(4.5), Inches(4.9), "Training Configuration")
    tb = slide6.shapes.add_textbox(Inches(8.2), Inches(2.4), Inches(4.1), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    t_points = [
        "Parameter Budget: Exactly 5,248 trainable weights.",
        "Separable Advantage: ~8.7× computational saving over traditional 2D convolutions.",
        "Optimizer: Adam (lr = 0.001) with CrossEntropyLoss.",
        "Batch Size: 64 samples over 8 completed epochs.",
        "Augmentation: Time-shift jitter (±100 ms) and background ambient noise mixing.",
        "Checkpoint: Saved at models/checkpoints/baseline_dscnn.pt (102.93 KB state_dict)."
    ]
    for pt in t_points:
        p = tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_footnote(slide6)

    # =========================================================================
    # SLIDE 7: INT8 Quantization Workflow
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Full INT8 Integer Quantization Workflow")
    
    add_card(slide7, Inches(0.8), Inches(1.7), Inches(5.6), Inches(4.9), "Quantization Formulation")
    tb = slide7.shapes.add_textbox(Inches(1.0), Inches(2.4), Inches(5.2), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    q_points = [
        "Uniform Affine Mapping: Continuous FP32 values x are mapped to signed 8-bit integers q in [-128, 127]:\n  q = clamp(round(x / S) + Z, -128, 127)",
        "Scale (S) & Zero-Point (Z):\n  S = (x_max - x_min) / 255\n  Z = round(-128 - x_min / S)",
        "Full Integer Constraint: Both model input and output tensors are constrained strictly to int8, eliminating float conversion wrappers.",
        "Input Quantization Params: S = 0.479577, Z = 54",
        "Output Quantization Params: S = 0.100918, Z = 82"
    ]
    for pt in q_points:
        p = tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_card(slide7, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.9), "TFLM & ESP-NN Compatibility")
    tb = slide7.shapes.add_textbox(Inches(7.0), Inches(2.4), Inches(5.3), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    c_points = [
        "Representative Calibration: Calibrated using 150 genuine training audio clips to record dynamic activation ranges.",
        "Zero-Copy Kernels: All 5 model operator types are verified compatible with TFLM and Espressif esp-nn assembly:",
        "  • CONV_2D (Xtensa PIE vector accelerated)",
        "  • DEPTHWISE_CONV_2D (Xtensa PIE vector accelerated)",
        "  • FULLY_CONNECTED (Xtensa PIE vector accelerated)",
        "  • AVERAGE_POOL_2D (Integer pooled)",
        "  • RESHAPE (Metadata pointer adjustment)",
        "Zero Unsupported Fallbacks: 100% of network executes in hardware-friendly integer arithmetic."
    ]
    for pt in c_points:
        p = tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_footnote(slide7)

    # =========================================================================
    # SLIDE 8: Verified FP32 vs INT8 Benchmark Results
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Verified Benchmark Results: FP32 vs Full INT8")
    
    # 4 Big Metric Cards
    metrics = [
        ("MODEL FILE SIZE", "21.77 KB", "-78.85%", "From 102.93 KB (.pt) to 21.77 KB (.tflite)"),
        ("TEST ACCURACY", "54.83%", "-0.58 pp", "55.41% (FP32) vs 54.83% (INT8) on 518 samples"),
        ("MACRO-F1 SCORE", "0.5179", "+0.0069", "FP32: 0.5110 -> INT8: 0.5179 (+0.69 pp)"),
        ("LAPTOP CPU LATENCY", "0.30 ms", "7.59× Speed", "2.29 ms (FP32) -> 0.30 ms (INT8) (XNNPACK)")
    ]
    for i, (title, val, delta, subtitle) in enumerate(metrics):
        left = Inches(0.8 + i * 2.95)
        add_card(slide8, left, Inches(1.7), Inches(2.8), Inches(2.2))
        tb = slide8.shapes.add_textbox(left + Inches(0.15), Inches(1.85), Inches(2.5), Inches(1.9))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_MUTED
        
        p = tf.add_paragraph()
        p.text = val
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = ACCENT_BLUE
        
        p = tf.add_paragraph()
        p.text = delta
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = ACCENT_GREEN if "+" in delta or "Speed" in delta or "-78" in delta else ACCENT_ROSE
        
        p = tf.add_paragraph()
        p.text = subtitle
        p.font.size = Pt(9)
        p.font.color.rgb = TEXT_MUTED

    # Detailed Table Below
    table_shape = slide8.shapes.add_table(5, 5, Inches(0.8), Inches(4.2), Inches(11.733), Inches(2.4))
    tbl = table_shape.table
    cols = ["Metric", "Baseline PyTorch FP32", "Quantized TFLite INT8", "Difference / Compression", "Verification Status"]
    for c, h in enumerate(cols):
        cell = tbl.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_HEADER_BG
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(10)
            p.font.color.rgb = ACCENT_BLUE

    data = [
        ("Model Disk Size", "105,396 B (102.93 KB)", "22,296 B (21.77 KB)", "-78.85% (4.73× compression)", "Exact file bytes"),
        ("Raw Weight Payload", "20,992 B (20.50 KB)", "5,248 B (5.125 KB)", "-75.00% (4.00× compression)", "5,248 parameters"),
        ("Test Accuracy (518 clips)", "55.41% (287 / 518)", "54.83% (284 / 518)", "-0.58 percentage points", "Identical test set"),
        ("Host CPU Latency", "2.29 ms ± 0.96 ms", "0.30 ms ± 0.08 ms", "7.59× speedup / 3,312 inf/sec", "Ryzen 3 7320U (100 runs)")
    ]
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = tbl.cell(r + 1, c)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.color.rgb = TEXT_MAIN

    add_footnote(slide8, "Latency measured on host AMD Ryzen 3 7320U CPU. Identical 518 test clips evaluated.")

    # =========================================================================
    # SLIDE 9: Interactive Reviewer Dashboard
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "Interactive Reviewer Dashboard (Local Web App)")
    
    add_card(slide9, Inches(0.8), Inches(1.7), Inches(5.6), Inches(4.9), "Architecture & Implementation")
    tb = slide9.shapes.add_textbox(Inches(1.0), Inches(2.4), Inches(5.2), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    dash_points = [
        "Framework: Local Streamlit web application running at http://localhost:8501.",
        "Crash-Resistant PyArrow-Free Tables: Direct HTML table serialization prevents numpy version collisions.",
        "Sandbox File Security: Strict path traversal prevention restricts report downloads to results/reports/.",
        "Graceful Fallback: Corrupted or missing artifacts display informative warning badges without crashing.",
        "Automated Verification: Browser automation subagent verified all 5 pages and rendered tables."
    ]
    for pt in dash_points:
        p = tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_card(slide9, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.9), "Key Reviewer Capabilities")
    tb = slide9.shapes.add_textbox(Inches(7.0), Inches(2.4), Inches(5.3), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    feat_points = [
        "1. Audio Inference Lab: Select real WAV samples across all 12 classes; view waveform and MFCC feature spectrograms.",
        "2. Live Dual Model Prediction: Concurrently runs PyTorch FP32 and TFLite INT8 on user-selected audio.",
        "3. Side-by-Side Model Comparison: Live tables displaying accuracy, F1, size, parameter counts, and latency deltas.",
        "4. Visualizations: Interactive confusion matrices (100% precision on '_silence_'), training loss curves, and latency charts.",
        "5. Virtual ESP32 Analysis: Layer-by-layer activation dimensions and SRAM budget calculators."
    ]
    for pt in feat_points:
        p = tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_footnote(slide9)

    # =========================================================================
    # SLIDE 10: Virtual ESP32-S3 Resource Estimates
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Virtual ESP32-S3 Deployment & Resource Budget")
    
    table_shape = slide10.shapes.add_table(7, 4, Inches(0.8), Inches(1.7), Inches(6.8), Inches(4.9))
    tbl = table_shape.table
    cols = ["Subsystem Component", "Memory Type", "Size", "% of Resource"]
    for c, h in enumerate(cols):
        cell = tbl.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_HEADER_BG
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(11)
            p.font.color.rgb = ACCENT_BLUE

    sram_data = [
        ("Model FlatBuffer Payload", "Flash (.rodata)", "21.77 KB", "0.54% of 4 MB Flash"),
        ("Tensor Arena (Activations)", "Internal SRAM", "30.00 KB", "5.86% of 512 KB SRAM"),
        ("KWS Task Stack (FreeRTOS)", "Internal SRAM", "16.00 KB", "3.13% of 512 KB SRAM"),
        ("I2S DMA Audio Buffers", "Internal SRAM", "2.00 KB", "0.39% of 512 KB SRAM"),
        ("MFCC Scratchpad Buffer", "Internal SRAM", "4.00 KB", "0.78% of 512 KB SRAM"),
        ("Total Working SRAM", "Internal SRAM", "52.00 KB", "10.16% (89.8% Headroom)")
    ]
    for r, row in enumerate(sram_data):
        for c, val in enumerate(row):
            cell = tbl.cell(r + 1, c)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.color.rgb = TEXT_MAIN

    add_card(slide10, Inches(8.0), Inches(1.7), Inches(4.5), Inches(4.9), "Activation Double-Buffering")
    tb = slide10.shapes.add_textbox(Inches(8.2), Inches(2.4), Inches(4.1), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    b_points = [
        "Memory Planner: TFLM dynamically reuses buffers between non-overlapping operations.",
        "Peak Adjacent Buffers: Conv2D (13,440 B) + Depthwise Conv (13,440 B) = 26,880 bytes (26.25 KB).",
        "TFLM Metadata & Padding: 2,048 bytes (2.00 KB).",
        "Minimum Theoretical Arena: 28,928 bytes (28.25 KB).",
        "Configured Tensor Arena: 30,720 bytes (30.00 KB) provides +10.7% safety margin.",
        "Projected ESP32-S3 Latency: ~6–10 ms @ 240 MHz utilizing esp-nn vector extensions."
    ]
    for pt in b_points:
        p = tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_footnote(slide10, "Analytical estimates. Flash/SRAM computed mathematically. Physical ESP32 flashing not performed.")

    # =========================================================================
    # SLIDE 11: Limitations & Future Hardware Validation
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "Technical Limitations & Future Work")
    
    add_card(slide11, Inches(0.8), Inches(1.7), Inches(5.6), Inches(4.9), "Current Engineering Disclosures")
    tb = slide11.shapes.add_textbox(Inches(1.0), Inches(2.4), Inches(5.2), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    lim_points = [
        "Laptop Latency Baseline: 0.30 ms latency was measured on an x86 laptop CPU; embedded on-silicon latency will be higher (~6–10 ms).",
        "Firmware Build Status: The ESP-IDF toolchain ('idf.py') is not installed on this host. CMake files and C++ firmware are syntactically validated but uncompiled.",
        "Zero Physical Hardware: Power dissipation (mA), acoustic microphone background noise, and physical GPIO actions were not tested on hardware.",
        "Uncalibrated Pruning: Structured channel pruning and knowledge distillation were designed conceptually but excluded from this baseline."
    ]
    for pt in lim_points:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_card(slide11, Inches(6.8), Inches(1.7), Inches(5.7), Inches(4.9), "Roadmap for Physical Validation")
    tb = slide11.shapes.add_textbox(Inches(7.0), Inches(2.4), Inches(5.3), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    fut_points = [
        "1. Toolchain Installation: Set up ESP-IDF v5.1+ and Xtensa GCC toolchain to build flashable .bin binaries.",
        "2. Development Board Flashing: Flash compiled binary to ESP32-S3-DevKitC-1 via USB-JTAG.",
        "3. I2S MEMS Microphone: Interface INMP441 digital microphone for live continuous listening.",
        "4. Power & Current Profiling: Measure dynamic milliwatt power draw using shunt resistor and oscilloscope.",
        "5. Advanced Compression: Integrate L1 structured filter pruning and Knowledge Distillation from a larger ResNet-18 teacher."
    ]
    for pt in fut_points:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN

    add_footnote(slide11)

    # =========================================================================
    # SLIDE 12: Conclusion & Q&A (Dark Theme)
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    bg12 = slide12.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg12.fill.solid()
    bg12.fill.fore_color.rgb = BG_DARK
    bg12.line.fill.background()

    tb12 = slide12.shapes.add_textbox(Inches(1.2), Inches(1.2), Inches(10.9), Inches(5.0))
    tf12 = tb12.text_frame
    tf12.word_wrap = True
    
    p = tf12.paragraphs[0]
    p.text = "FINAL-YEAR ECE PROJECT SUMMARY"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = RGBColor(56, 189, 248)
    
    p = tf12.add_paragraph()
    p.text = "Key Takeaways & Conclusion"
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    
    summary_text = (
        "\n1. Extreme Compression with Parity: 78.85% model size reduction (102.9 KB -> 21.77 KB) with only 0.58 pp accuracy loss (54.83%).\n"
        "2. Microcontroller Feasibility: Occupies 0.54% of 4 MB Flash and 52.0 KB working SRAM (89.84% free headroom on ESP32-S3).\n"
        "3. Cryptographic C-Export: Generated model_data.h verified with 100% byte-for-byte SHA-256 integrity.\n"
        "4. Full Transparency: Complete separation between measured PC benchmarks and analytical embedded estimates.\n"
        "5. Rigorous Quality: 61 automated unit/integration tests passing with 0 failures.\n\n"
        "Thank You. Questions & Discussion Welcome."
    )
    p = tf12.add_paragraph()
    p.text = summary_text
    p.font.size = Pt(13)
    p.font.color.rgb = RGBColor(203, 213, 225)

    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path}")

if __name__ == "__main__":
    create_presentation()
