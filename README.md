# AQG Bloom-Conditioned Question Generation

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.46+-yellow)](https://huggingface.co/docs/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Bloom-Conditioned Question Generation with T5 and DistilBERT Auto-Labeling**  
> Submitted to IEEE ICET 2026 (21st International Conference on Emerging Technologies)  
> GIKI, Topi, Pakistan — November 18-19, 2026

## Overview

This repository contains a complete pipeline for generating cognitively-controlled questions across all six levels of Bloom's Taxonomy using a fine-tuned T5-base model. The system uses a Bloom-level control token that allows the same model to generate questions targeting different cognitive levels at inference time.

## Key Features

- **Three-stage Auto-Labeler:** V1 keyword (76%) → V2 regex (86%) → V3 DistilBERT (97%)
- **Cognitive Distribution Analysis:** Empirical analysis of SQuAD, SciQ, and RACE
- **Balanced Training Corpus:** 7,300 RACE samples across all 6 Bloom levels
- **Bloom-Conditioned T5:** Single model generates questions for all 6 levels via control tokens
- **Per-Level Evaluation:** Detailed BLEU-4 and ROUGE metrics for each Bloom level
- **Reproducible:** Runs end-to-end on a single T4 GPU (Google Colab free tier)

## Results

### Auto-Labeler Accuracy (100 manually labeled samples)

| Dataset | V1 (Keyword) | V2 (Regex) | V3 (DistilBERT) |
|---------|--------------|------------|-----------------|
| SQuAD (n=35) | 94.3% | 97.1% | 97.1% |
| SciQ (n=35) | 85.7% | 91.4% | 94.3% |
| RACE (n=30) | 43.3% | 66.7% | **100.0%** |
| **Overall (n=100)** | **76.0%** | **86.0%** | **97.0%** |

### Question Generation Metrics (200 validation samples)

| Bloom Level | N | BLEU-4 | ROUGE-1 | ROUGE-L |
|-------------|---|--------|---------|---------|
| Remember | 55 | 9.74 | 31.27 | 29.41 |
| Understand | 53 | 23.44 | 49.70 | 49.04 |
| Apply | 3 | 11.97 | 39.17 | 37.27 |
| **Analyze** | 63 | **30.91** | **55.49** | **53.43** |
| Evaluate | 21 | 13.62 | 41.16 | 35.85 |
| Create | 5 | 9.10 | 32.69 | 32.69 |
| **Overall** | **200** | **20.46** | **44.98** | **43.05** |

## Project Structure

```
AQG_Bloom_Project/
├── AQG_Bloom_Project_CLEAN.ipynb   # Main notebook (run in Colab)
├── README.md                        # This file
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
└── docs/                            # Documentation and paper materials
    ├── paper.pdf                    # Conference paper
    └── figures/                     # Result figures
```

## Quick Start

### Option 1: Google Colab (Recommended)

1. Click here to open in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Muzammal-Bilal/AQG_Bloom_Project/blob/main/AQG_Bloom_Project_CLEAN.ipynb)
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run cells sequentially (mount Google Drive when prompted)

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/Muzammal-Bilal/AQG_Bloom_Project.git
cd AQG_Bloom_Project

# Install dependencies
pip install -r requirements.txt

# Open notebook
jupyter notebook AQG_Bloom_Project_CLEAN.ipynb
```

## Pipeline

The notebook follows a clear 15-section structure:

1. **Setup & Dependencies** — Install libraries, mount Google Drive
2. **Load Datasets** — SQuAD, SciQ, RACE from HuggingFace
3. **Auto-Labeler V1** — Keyword-based classifier (76% accuracy)
4. **Auto-Labeler V2** — Regex pattern classifier (86% accuracy)
5. **Cognitive Distribution Analysis** — Bloom-level breakdown per dataset
6. **Save Labeled Data** — Persist all 187K labeled questions
7. **Build Balanced Training Set** — 7,300 RACE samples across 6 levels
8. **Prepare T5 Training Format** — Input/output formatting
9. **T5-base Model Setup** — Load pre-trained T5
10. **Fine-tune T5** — 3 epochs, AdamW, lr=3e-5 (~47 min on T4)
11. **Test Question Generation** — Sample questions across all levels
12. **Evaluation** — BLEU-4, ROUGE-1/2/L on 200 validation samples
13. **Auto-Labeler V3** — Fine-tune DistilBERT (97% accuracy)
14. **Generate Paper Figures** — Save evaluation results
15. **Project Files Summary** — Output files documentation

## Hardware Requirements

| Resource | Required |
|----------|----------|
| GPU | T4 or better (free on Colab) |
| RAM | 12 GB+ |
| Storage | 3 GB on Google Drive |
| Total runtime | ~2 hours |

## Datasets

| Dataset | Source | Size | License |
|---------|--------|------|---------|
| SQuAD v1.1 | [HuggingFace](https://huggingface.co/datasets/rajpurkar/squad) | 87,599 | CC BY-SA 4.0 |
| SciQ | [HuggingFace](https://huggingface.co/datasets/allenai/sciq) | 11,679 | CC BY-NC 3.0 |
| RACE | [HuggingFace](https://huggingface.co/datasets/ehovy/race) | 87,866 | Research-only |

## Models Used

- **T5-base** (220M parameters) — [HuggingFace](https://huggingface.co/t5-base)
- **DistilBERT-base-uncased** (66M parameters) — [HuggingFace](https://huggingface.co/distilbert-base-uncased)

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{bilal2026aqg,
  author    = {Bilal, Muzammal and Taseer, Suleman},
  title     = {Bloom-Conditioned Question Generation with T5 and DistilBERT Auto-Labeling},
  booktitle = {Proceedings of the 21st International Conference on Emerging Technologies (ICET)},
  year      = {2026},
  organization = {IEEE},
  address   = {GIKI, Topi, Pakistan},
  month     = {November}
}
```

## Authors

- **Muzammal Bilal** — MS in Artificial Intelligence, Bahria University Lahore  
  Reg# 03-205252-007
- **Dr. Suleman Taseer** — Supervisor, Bahria University Lahore

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- HuggingFace for transformers library and dataset hosting
- Google Colab for free GPU access
- Bahria University Lahore for academic support

## Contact

For questions or collaborations:
- Email: MUZAMMALBILAL36@gmail.com
- Issues: [Open an issue](https://github.com/Muzammal-Bilal/AQG_Bloom_Project/issues)

---

**Last Updated:** April 2026
