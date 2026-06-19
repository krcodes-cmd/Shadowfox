# Shadowfox — Unified AI/ML Model Hub

Welcome to the **Unified AI Model Hub**, a premium AI/ML portfolio project showcasing high-performance deep learning models for regression, multi-task classification, and Natural Language Processing (NLP). 

Developed by **Krisha Patel**, this application integrates three distinct production-grade models into a single, interactive Streamlit dashboard.

---

## 🌟 Features

### 🏘️ T1: Boston Housing Predictor (Regression)
A lightweight residual Multi-Layer Perceptron (MLP) designed to predict median home values based on neighborhood characteristics.
- **Architecture:** Residual MLP with BatchNorm, GELU activation, and Dropout.
- **Performance:** Achieves **R² > 0.80**.
- **Tech Stack:** PyTorch, Scikit-Learn.
- **Highlights:** Feature engineering (interactions like RM², LSTAT²), log-transform target stabilization, and OneCycleLR scheduling for fast convergence.

### 📊 T2: NEXUS-RT v1 — Sales Analytics (Dual-Head Classification)
A sophisticated dual-head residual network for simultaneous multi-task prediction on retail transaction data (Superstore dataset).
- **Architecture:** Dual-head Residual MLP with 46 engineered features.
- **Head A (Binary):** Profitability Prediction (**~88% Accuracy**).
- **Head B (Multi-class):** Product Sub-Category Classification (**~99.75% Accuracy**).
- **Tech Stack:** PyTorch, Pandas.
- **Highlights:** Simultaneous inference for strategic business insights.

### 💬 T3: NLP Studio (BERT & GPT-2)
A comprehensive NLP playground exploring both encoder (BERT) and decoder (GPT-2) transformer architectures.
- **Sentiment Analysis:** BERT-base-uncased fine-tuned on SST-2 for high-precision sentiment classification (**≥ 93% Accuracy**).
- **Text Generation:** GPT-2 for controllable, domain-adaptive text generation with adjustable temperature, top-p, and top-k sampling.
- **Tech Stack:** HuggingFace Transformers, PyTorch.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- (Optional) CUDA-enabled GPU for faster inference

### Installation & Launch

The project includes a convenient launcher script that automatically handles dependency installation and starts the application.

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Final
   ```

2. **Run the launcher:**
   ```bash
   python run.py
   ```

Alternatively, you can install the dependencies manually and run Streamlit:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```text
Final/
├── app.py                # Main Streamlit application
├── run.py                # Launcher & dependency installer
├── requirements.txt      # Project dependencies
├── T1/                   # Boston Housing Model & Dataset
├── T2/                   # NEXUS-RT v1 Model & Dataset
├── T3/                   # NLP Research & Notebooks
├── utils/                # Modular inference logic for app.py
└── .streamlit/           # Streamlit configuration
```

---

## 🛠️ Technical Specifications

| Model | Architecture | Parameters | Task |
| :--- | :--- | :--- | :--- |
| **BostonHousingNet** | Residual MLP | ~20K | Regression |
| **NEXUS-RT v1** | Dual-Head MLP | 174,515 | Multi-Task Classification |
| **BERT-base** | Transformer Encoder | 110M | Sentiment Analysis |
| **GPT-2** | Transformer Decoder | 124M | Text Generation |

---

## 👤 Author & Creator

**Krisha Patel**  
*AI/ML Enthusiast & Developer*  

Built with ❤️ using Streamlit, PyTorch, and HuggingFace Transformers.
