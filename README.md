# 🩺 MedShield AI — Adversarially Robust Generative AI

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medshield-ai.streamlit.app)

> Trains a GAN on chest X-ray images, attacks it with adversarial techniques, defends against those attacks, classifies X-rays as **NORMAL** or **PNEUMONIA**, and explains model decisions using Grad-CAM — all through a professional interactive interface.

[![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📌 What This Project Does

1. **Trains a DCGAN** — learns to generate realistic fake chest X-rays from random noise
2. **Trains a CNN Classifier** — distinguishes NORMAL lungs from PNEUMONIA cases
3. **Attacks the model** — FGSM and PGD add invisible noise to fool the AI
4. **Defends against attacks** — Gaussian denoising and median filter clean the image
5. **Explains decisions** — Grad-CAM heatmaps show where the model looks
6. **Confusion matrix** — evaluates classifier performance on the test set
7. **Interactive interface** — full pipeline accessible through a 5-tab Streamlit app with dark/light theme

---

## 🗂️ Project Structure

```
adversarial_ai_project/
│
├── app.py                     ← Main Streamlit interface (run this)
│
├── model/
│   ├── __init__.py
│   ├── gan.py                 ← DCGAN: Generator + Discriminator
│   ├── train.py               ← GAN training loop + save/load functions
│   ├── classifier.py          ← 4-block CNN classifier architecture
│   └── train_classifier.py   ← Classifier training + evaluation
│
├── attacks/
│   ├── __init__.py
│   ├── fgsm.py                ← Fast Gradient Sign Method
│   └── pgd.py                 ← Projected Gradient Descent
│
├── defense/
│   ├── __init__.py
│   └── defend.py              ← Denoising + adversarial detection
│
├── xai/
│   ├── __init__.py
│   └── gradcam.py             ← Grad-CAM heatmap + saliency maps
│
├── utils/
│   ├── __init__.py
│   └── data_loader.py         ← Chest X-ray dataset loader
│
├── requirements.txt
├── LICENSE
└── README.md
```

> **Note:** `data/`, `*.pth` model weights, and `venv/` are excluded from the repo via `.gitignore`.

---

## 🧠 How It Works

### GAN (Generative Adversarial Network)
The **Generator** creates fake chest X-rays from 100-dimensional random noise using `ConvTranspose2d` layers. The **Discriminator** tries to tell real X-rays from fake ones using `Conv2d` layers. They compete until the Generator produces convincing fakes — and the trained Discriminator becomes the adversarial attack target.

### Adversarial Attacks
Invisible noise added to an image at pixel level that the human eye cannot detect, but causes the model to make a completely wrong prediction.
- **FGSM** — one large gradient step scaled by ε. Fast and simple.
- **PGD** — many small iterative steps projected back into the ε-ball. The strongest first-order attack.

### Defense
Pre-processing applied before the image reaches the classifier:
- **Gaussian denoising** — disrupts structured adversarial perturbations
- **Median filter** — average pooling removes pixel-level noise
- **Detection** — flags images with suspicious confidence drops after smoothing

### Explainability (Grad-CAM)
Shows which regions of the X-ray the model focuses on when making its decision. Red/yellow = high attention. Adversarial attacks visibly shift this attention away from the lung region.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.9 | Core language |
| PyTorch 2.x | All model training and inference |
| Streamlit | Interactive 5-tab web interface |
| Matplotlib | Loss curves, confusion matrix, training charts |
| pytorch-grad-cam | Grad-CAM heatmap visualisation |
| Pillow | Image loading and preprocessing |
| scikit-learn | Evaluation utilities |
| Apple MPS (M4) | GPU acceleration on Apple Silicon |

---

## 📦 Installation

### Step 1 — Clone the repo
```bash
git clone https://github.com/himansh1241/adversarial-robust-generative-ai.git
cd adversarial-robust-generative-ai
```

### Step 2 — Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Download the dataset
Go to 👉 [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

After downloading, place the unzipped folder so your structure looks like:
```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    └── test/
```

### Step 5 — Run the app
```bash
streamlit run app.py
```

Your browser opens at `http://localhost:8501` automatically.

---

## 🚀 How to Use

Follow the tabs in order. Each step builds on the previous one.

| Tab | What to do | What you see |
|---|---|---|
| 📊 Train GAN | Set epochs → click Start Training | Loss curves + generated X-ray samples |
| 🫁 Classifier | Set epochs → click Train Classifier | Accuracy curve + confusion matrix |
| ⚔️ Attacks | Choose FGSM / PGD → click Run Attack | Score shift + adversarial image |
| 🛡️ Defense | Choose method → click Apply Defense | Before/after + detection result |
| 📂 Diagnose | Upload X-ray → click Run Pipeline | NORMAL/PNEUMONIA + Grad-CAM heatmap |

**Tip:** After training once, use the **Load Saved GAN** and **Load Saved Classifier** buttons to reload models instantly without retraining. Model weights are saved automatically as `generator.pth`, `discriminator.pth`, and `classifier.pth`.

---

## ✨ Interface Features

- **Dark / light theme toggle** — switch anytime without losing training progress
- **Pipeline progress bar** — shows which steps are complete and what to do next
- **Sidebar val accuracy gauge** — live accuracy bar updates after classifier training
- **Load saved models** — skip retraining by loading weights from disk
- **Live device badge** — shows whether Apple MPS, CUDA, or CPU is being used
- **Confusion matrix** — TP/FP/TN/FN breakdown on the test set after training
- **Grad-CAM heatmap** — visual attention map overlaid on uploaded X-ray
- **Tab fade-in animation** — smooth tab transitions throughout

---

## 📊 Results

| Metric | Value |
|---|---|
| Classifier test accuracy | ~80% |
| FGSM attack score shift | +0.31 average |
| PGD 40-step accuracy drop | ~48% (near coin-flip) |
| Defense recovery | Back to ~79% with combined method |
| Attack detection rate | 100% with confidence-drop method |

**Key finding:** An invisible perturbation (ε = 0.1) was sufficient to flip the diagnosis from NORMAL to PNEUMONIA in an undefended model. After applying combined defense, the correct diagnosis was restored.

---

## 🔑 Glossary

| Term | Plain meaning |
|---|---|
| GAN | Two AIs competing — one fakes images, one detects fakes |
| DCGAN | GAN using convolutional layers for better image quality |
| FGSM | One-step attack along the gradient direction |
| PGD | Multi-step iterative attack — strongest first-order method |
| Epsilon (ε) | Attack strength — 0.01 is weak, 0.5 is strong |
| Grad-CAM | Heatmap showing which pixels influenced the model's decision |
| MPS | Apple Silicon GPU backend in PyTorch |

---

## ⚠️ Disclaimer

This project is **for educational and research purposes only**. The classifier is not validated for clinical use and must not be used for real medical diagnosis.

---

## 👤 Author

**Himanshu Ranjan**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/himanshuranjan1241)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/himansh1241)

---

## 📜 License

This project is open source under the [MIT License](LICENSE).

---

<div align="center">
    <sub>Built with PyTorch · Streamlit · Apple Silicon M4 · For educational use only</sub>
</div>
