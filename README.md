# BARI 72 & 73 Potato Leaves Disease Classification

[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework: TensorFlow / Keras](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange.svg)](https://www.tensorflow.org/)
[![Thesis: M.Sc. ICE](https://img.shields.io/badge/Degree-M.Sc.%20Thesis%20(PDF)-red.svg)](./Msc(ICE)%20Jesmin%20Akther.pdf)
[![Paper: IJATEE 2024](https://img.shields.io/badge/Paper-IJATEE%202024-green.svg)](https://doi.org/10.19101/IJATEE.2024.111100083)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

This repository contains the research codebase, machine learning models, and official Master of Science (M.Sc.) thesis documentation for the automated diagnosis and classification of potato leaf blight diseases affecting **BARI-72** and **BARI-73** potato cultivars in Bangladesh using deep learning and **Simplistic Convolutional Neural Networks (SCNN)**.

---

## 📄 Master's Thesis Documentation

The full, official Master of Science in Information and Communication Engineering (ICE) thesis report is available directly in this repository:

* **Document:** [**Msc(ICE) Jesmin Akther.pdf**](./Msc(ICE)%20Jesmin%20Akther.pdf)
* **Author:** **Jesmin Akther**
* **Institution:** Department of Information and Communication Engineering, **Noakhali Science and Technology University (NSTU)**, Bangladesh
* **Honors & Fellowships:**
  * **National Science and Technology (NST) Fellowship**, Ministry of Science and Technology, Government of Bangladesh (2021)
  * **Research Cell Fellowship for Masters Research**, Noakhali Science and Technology University (2021)

### 📚 Associated Peer-Reviewed Publications

1. **A. Khan, J. Akther, and F. Rahman**  
   *"Comparative analysis of potato blight diseases BARI-72 and BARI-73 using a simplified convolutional neural network method."*  
   **International Journal of Advanced Technology and Engineering Exploration (IJATEE)**, Vol. 11, Issue 115, pp. 819–837, 2024.  
   DOI: [10.19101/IJATEE.2024.111100083](https://doi.org/10.19101/IJATEE.2024.111100083)

2. **J. Akther, M. Harun-or-roshid, and A. A. Nayan**  
   *"Potato Leaves Blight Disease Classification in Bangladesh Perspective Using Convolutional Neural Network (CNN)."*  
   **Engineering Journal**, Vol. 27, No. 7, pp. 27–38, 2023.

3. **J. Akther, M. Harun-or-roshid, A. A. Nayan, and M. G. Kibria**  
   *"Transfer learning on VGG16 for the classification of potato leaves infected by blight diseases."*  
   **IEEE Emerging Technology in Computing, Communication and Electronics (ETCCE)**, pp. 1–5, 2021.

---

## 🔬 Research Background & Motivation

Potato is one of the most vital agricultural food crops and economic staples in Bangladesh. However, foliar diseases—specifically **Early Blight** (*Alternaria solani*) and **Late Blight** (*Phytophthora infestans*)—frequently cause catastrophic yield reductions across major cultivars, including **BARI Alu-72** and **BARI Alu-73** developed by the Bangladesh Agricultural Research Institute (BARI).

### Key Challenges Addressed:
* **Computational Overhead:** Conventional deep convolutional neural networks (ResNet, VGG, DenseNet) require massive parameter counts, large memory footprints, and extensive training time.
* **Low-Resource Deployment:** Agricultural field operations require fast, lightweight models capable of inference on mobile or edge devices without expensive GPU hardware.
* **Cultivar-Specific Nuances:** Comparative evaluation across specific localized potato varieties (BARI-72 vs. BARI-73) under real-world field conditions.

---

## 🧠 Methodology & Architectures

The implemented framework explores both standard convolutional neural networks and lightweight architectures:

1. **Simplistic Convolutional Neural Network (SCNN):**
   * Engineered with progressive hidden layers (e.g., 16, 16, 32 kernels / 256 to 64 filter reductions).
   * Incorporates specialized dropout, batch normalization, and L2 regularization to prevent overfitting on localized datasets.
   * Delivers **95.69%–96.09% classification accuracy** while drastically slashing model parameter count and inference latency.
2. **Transfer Learning Benchmarks:**
   * Pretrained feature extraction using VGG16, ResNet, and EfficientNet architectures to establish baseline performance metrics.
3. **Hyperparameter Tuning via TensorBoard:**
   * Dynamic monitoring of training loss, validation loss, validation accuracy, and gradient distributions across epochs.

---

## 💻 Source Code Overview

The core model definition and training pipeline is provided in:

* [`Bari72&73_potato.py`](./Bari72&73_potato.py)

### Pipeline Stages in the Script:
1. **Data Ingestion & Preprocessing:** Loads image directories, performs RGB-to-grayscale conversion, and resizes input samples to standardized dimensions (e.g., 50×50 or 100×100 pixels).
2. **Normalization & Pickling:** Scales pixel intensities (`X / 255.0`) and serializes preprocessed feature matrices (`w1.pickle`) and label vectors (`z1.pickle`).
3. **Model Construction:** Configures progressive convolutional blocks:
   * `Conv2D(256, (3, 3))` + `ReLU` + `MaxPooling2D(2, 2)`
   * `Conv2D(64, (3, 3))` + `ReLU` + `MaxPooling2D(2, 2)`
   * `Flatten()` + `Dense(64)` + `Dense(1, activation='sigmoid')`
4. **Optimization:** Compiles with the **Adam** optimizer, `binary_crossentropy` loss, and integrates `TensorBoard` callback logging.
5. **Evaluation:** Validates on a 30% held-out test split, tracking convergence behavior and confusion metrics.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8, 3.9, or 3.10
* Virtual environment (recommended)

### Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/JesminNipu/BARI-72-73-Potato-Leaves-Disease.git
   cd BARI-72-73-Potato-Leaves-Disease
   ```

2. **Install dependencies:**
   ```bash
   pip install tensorflow opencv-python numpy matplotlib tqdm
   ```

3. **Run the training script:**
   ```bash
   python Bari72&73_potato.py
   ```

4. **Launch TensorBoard (Optional):**
   ```bash
   tensorboard --logdir=errorr1/
   ```

---

## 📂 Repository Structure

```text
BARI-72-73-Potato-Leaves-Disease/
├── Bari72&73_potato.py            # Model architecture, preprocessing, and training pipeline
├── Msc(ICE) Jesmin Akther.pdf     # Official M.Sc. thesis document (converted from DOCX)
├── README.md                      # Comprehensive academic and technical documentation
└── .gitignore                     # Python, cache, and checkpoint ignore rules
```

---

## 👩‍💻 Author & Contact

**Jesmin Akther**  
Mobile Application Developer & Computer Science Researcher  
* Department of Information and Communication Engineering (ICE), NSTU  
* **Email:** [jesminnipu1@gmail.com](mailto:jesminnipu1@gmail.com)  
* **GitHub:** [@JesminNipu](https://github.com/JesminNipu)  
* **Academic Portfolio:** [https://jesminnipu.github.io/Portfolio/](https://jesminnipu.github.io/Portfolio/)
