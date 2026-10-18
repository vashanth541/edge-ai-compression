# Deploying Edge AI Compression Framework to Streamlit Community Cloud

This guide provides step-by-step instructions for deploying the **Edge AI Model Compression and Deployment Framework for ESP32-S3** to [Streamlit Community Cloud](https://share.streamlit.io/) via GitHub.

---

## Architecture & Deployment Overview

- **Framework**: Streamlit (`>=1.32.0`)
- **Python Version**: `3.11`
- **Application Entry Point**: `src/dashboard/app.py`
- **Primary Cloud Artifact**: `deployment/baseline_dscnn_int8_esp32s3_deploy.zip` (37,018 bytes)
- **Built-in Models**:
  - Quantized INT8: `models/quantized/baseline_dscnn_int8.tflite` (22,296 bytes)
  - Baseline FP32: `models/quantized/baseline_dscnn_fp32.tflite` (29,184 bytes)
  - PyTorch Checkpoint: `models/checkpoints/baseline_dscnn.pt` (105,797 bytes)

All required runtime dependencies are defined in [`requirements.txt`](./requirements.txt). Heavy training-only packages (e.g., raw torchaudio training loaders or audio file synthesizers) are excluded to ensure fast cloud container provisioning.

---

## Step 1: Review Local Git Changes

Open PowerShell in the project directory:
```powershell
cd C:\Users\ELCOT\OneDrive\Documents\edge-ai-compression
```

Verify your current status and ensure that temporary caches and virtual environments are ignored:
```powershell
git status
```

You should see only modified and untracked project files (such as `src/dashboard/app.py`, `src/dashboard/package_builder.py`, `requirements.txt`, `.gitignore`, `STREAMLIT_DEPLOYMENT.md`, and `deployment/baseline_dscnn_int8_esp32s3_deploy.zip`).

> **Note**: `.venv/`, `.pytest_cache/`, `scratch/`, and large raw audio datasets are automatically excluded by `.gitignore`.

---

## Step 2: Commit and Push to GitHub

Stage the modified files, commit them, and push to your remote GitHub repository.

### Option A: Using Git CLI in Terminal
```powershell
# 1. Stage all changes
git add .

# 2. Commit with a clear descriptive message
git commit -m "Prepare Edge AI Compression Framework for Streamlit Cloud deployment with 1-click INT8 ZIP download"

# 3. If you haven't linked a GitHub remote repository yet:
# git remote add origin https://github.com/<your-github-username>/edge-ai-compression.git
# git branch -M main

# 4. Push to GitHub
git push -u origin main
```

### Option B: Using GitHub Desktop or VS Code Git
1. Open the Source Control tab in VS Code (`Ctrl + Shift + G`).
2. Review staged changes.
3. Enter commit message: `Prepare Edge AI Compression Framework for Streamlit Cloud deployment`.
4. Click **Commit** and then **Sync Changes / Push**.

---

## Step 3: Sign In to Streamlit Community Cloud

1. Open your browser and navigate to **[share.streamlit.io](https://share.streamlit.io/)**.
2. Click **"Continue with GitHub"** to authenticate.
3. Grant Streamlit Cloud read permissions to your GitHub repositories if prompted.

---

## Step 4: Create a New Streamlit App

1. In your Streamlit Cloud workspace, click the **"Create app"** (or **"New app"**) button in the upper right.
2. Select **"I already have an app"** (or choose repository directly from GitHub).
3. Fill in the deployment configuration fields:
   - **Repository**: `<your-github-username>/edge-ai-compression`
   - **Branch**: `main` (or your active branch)
   - **Main file path**: `src/dashboard/app.py`
   - **App URL (optional)**: Custom subdomain (e.g., `edge-ai-esp32s3-compression.streamlit.app`)
4. Expand **Advanced Settings**:
   - **Python version**: Select `3.11`.
   - **Secrets**: Not required (no external API keys or cloud credentials needed for offline inference and deployment package generation).
5. Click **"Deploy!"**.

---

## Step 5: Monitor Startup Logs & Deployment

Streamlit Cloud will provision a Debian Linux container, install dependencies from `requirements.txt`, and launch `src/dashboard/app.py`.

- If the app takes a moment to build, click **"Manage app"** in the lower right corner to view the real-time terminal logs.
- Successful startup will display:
  ```text
  Collecting packages from requirements.txt...
  Successfully installed streamlit, tensorflow-cpu, torch...
  Running app.py...
  ```
- If an installation error occurs, verify that Python `3.11` is selected in App Settings.

---

## Step 6: Test the Public URL & Features

Once the app is running:
1. Open the public URL provided by Streamlit Cloud (e.g., `https://<your-subdomain>.streamlit.app`).
2. Verify the top hero section:
   - **Header**: `⚡ Download ESP32-S3 INT8 Deployment Package`
   - **Metrics**: Model Binary Size (`21.77 KB` / `22,296 B`), Compression Ratio (`1.31×` / `-23.60% vs FP32`), Test Accuracy (`54.83%`), Deployment Bundle (`36.2 KB` / `37,018 B`).
3. Expand **"📋 Package Contents & Verification Manifest (8 Artifacts)"** to verify that SHA-256 hashes and sizes are validated.
4. Navigate through sidebar sections:
   - **🔧 Compression Framework**: Test custom model inspection and ESP32-S3 static compatibility check.
   - **Audio Inference Lab**: Test real-time waveform and MFCC keyword classification.
   - **Model Comparison**: Review FP32 vs. INT8 pareto benchmarks.

---

## Step 7: Download and Verify the Deployment ZIP

1. Click the primary button: **"⬇️ Download ESP32-S3 INT8 Deployment Package (.zip)"**.
2. Save `baseline_dscnn_int8_esp32s3_deploy.zip` to your computer.
3. Open or extract the ZIP file and confirm that all 8 expected artifacts are present:
   - `models/baseline_dscnn_int8.tflite` (exact 22,296 bytes)
   - `include/model_data.h` (C array with `alignas(8)`)
   - `include/model_labels.h` (12 speech command classes)
   - `main.cpp` (ESP32-S3 TFLM inference template)
   - `compatibility_report.md` (static analysis: Flash 0.5%, SRAM 6.5%)
   - `deployment_report.json` (machine-readable metadata)
   - `README.md` (ESP-IDF build guide & disclaimer)
   - `MANIFEST.json` (cryptographic verification)
4. Confirm the archive extracts cleanly without corruption.

---

## Local Development & Troubleshooting

To run or debug the dashboard locally on Windows:
```powershell
.\.venv\Scripts\streamlit.exe run src/dashboard/app.py --server.port 8501
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Important Technical Disclosures
- **Static Verification vs. Physical Hardware**: Compatibility checker results indicate static memory and operator validation against ESP32-S3 limits (16 MB Flash, 512 KB SRAM). Physical on-device execution requires the ESP-IDF toolchain and an actual ESP32-S3 development board.
