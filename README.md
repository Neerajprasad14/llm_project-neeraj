# Cross-Lingual Abstractive Summarization (Indic Languages to English)

This repository contains an end-to-end deep learning pipeline for cross-lingual abstractive summarization. It fine-tunes the `ai4bharat/IndicBART` sequence-to-sequence model to ingest news articles and documents in **Tamil, Telugu, and Sanskrit**, and generates concise, abstractive summaries in **English**.

By leveraging 4-bit Quantization (QLoRA) and Parameter-Efficient Fine-Tuning (PEFT), this pipeline is optimized to run on consumer-grade hardware (like a Google Colab T4 GPU) while preventing catastrophic forgetting and maintaining high inference quality.

## 🚀 Key Features
* **Multi-Source Language Support:** Processes Tamil (`ta`), Telugu (`te`), and custom Sanskrit (`sa`) datasets.
* **Cross-Lingual Generation:** Forces decoder probability distributions entirely into English (`en`), successfully bypassing the "cross-lingual bleeding" common in mBART architectures.
* **Script Unification:** Utilizes `indic-nlp-library` to transliterate all Indic scripts into Devanagari before tokenization, aligning with IndicBART's native pre-training environment.
* **Memory-Efficient Training:** Implements `BitsAndBytes` `nf4` quantization and LoRA adapters, keeping VRAM usage under 10GB.
* **Custom Tokenization Handling:** Bypasses Hugging Face's default `[CLS]`/`[SEP]` injections (`add_special_tokens=False`) to preserve explicit language directionality tags (e.g., `</s> <2en>`).

## 🛠️ Tech Stack
* **Framework:** PyTorch
* **Transformers Library:** Hugging Face `transformers`, `datasets`, `evaluate`
* **Optimization:** `peft` (LoRA), `bitsandbytes` (4-bit Quantization)
* **NLP Tools:** `indic-nlp-library`
* **Model Base:** `ai4bharat/IndicBART`

## 📦 Installation

Ensure you have a GPU-enabled environment, then install the required dependencies:

```bash
pip install torch transformers datasets peft bitsandbytes evaluate accelerate
pip install indic-nlp-library sacrebleu rouge-score
