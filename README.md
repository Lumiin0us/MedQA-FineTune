# MedQA-FineTune — Medical QA Fine-Tuning with QLoRA

Fine-tuning SmolLM2-1.7B-Instruct on a medical question answering dataset using QLoRA. This project demonstrates the full fine-tuning pipeline and showing measurable improvement over the base model on medical domain questions.

---

## Results

|     Metric     | Base Model | Fine-Tuned |    Improvement    |
|----------------|------------|------------|-------------------|
| ROUGE-L        |    0.122   |    0.157   |      +28.9%       |
| Cases Improved |      —     |    37/50   | 74% of test cases |

---

## Dataset

**ChatDoctor-HealthCareMagic-100k** from HuggingFace — 100,000 real doctor-patient conversations from the HealthCareMagic platform.

I have taken a subset of this dataset:
  - 4,500 training samples
  - 500 validation samples
  - Filtered to remove empty or low quality entries
  - Formatted as instruction-response pairs

Each training example:
```
<|user|>
{patient question}
<|assistant|>
{doctor response}
```

---

## Method — QLoRA

Full fine-tuning of a 1.7B model requires significant GPU memory. QLoRA (Quantized Low Rank Adaptation) makes this feasible on consumer hardware by:

1. **Quantizing** the base model to 4-bit precision — reduces memory from ~7GB to ~2GB
2. **Adding LoRA adapters** — small trainable matrices inserted into attention layers
3. **Training only the adapters** — base model stays frozen


### LoRA Configuration
| Parameter | Value |
|-----------|-------|
|  Rank (r) |  16   |
|   Alpha   |  32   |
|  Dropout  | 0.05  |
| Target modules | q_proj, v_proj |

### Training Configuration
|    Parameter   |  Value   |
|----------------|----------|
|     Epochs     |     3    |
|   Batch size   |     4    |
|  Learning rate |   2e-4   |
|  LR scheduler  |  Cosine  |
|  Warmup ratio  |   0.03   |
|   Precision    | BFloat16 |

---

## Training Environment

Trained on **Google Colab T4 GPU** (16GB VRAM) using QLoRA quantization.

---

## Project Structure
```
MedQA-FineTune/
├── data/
│   ├── train.jsonl          ← 4,500 training samples
│   └── valid.jsonl          ← 500 validation samples
├── adapters/
│   └── final/               ← saved LoRA adapter weights
├── prepare_data.py          ← dataset download and formatting
├── train.py                 ← QLoRA fine-tuning script
├── evaluate.py              ← ROUGE evaluation
├── testInference.py         ← base vs fine-tuned comparison
├── requirements.txt
└── README.md
```

---

## Pipeline
```
ChatDoctor Dataset (HuggingFace)
        ↓
prepare_data.py
  → filter low quality entries
  → format as conversation pairs
  → split 90/10 train/validation
  → save as JSONL
        ↓
train.py
  → load SmolLM2-1.7B in 4-bit (QLoRA)
  → configure LoRA adapters
  → train with SFTTrainer
  → save adapters to disk
        ↓
evaluate.py
  → load base model + fine-tuned model
  → run 50 test cases through both
  → compute ROUGE-L scores
  → compare results
```

---

## Example Output

**Question:** I have been having chest pain for 2 days. What could it be?

**Base Model:**
> Chest pain can be caused by many things. Please see a doctor.

**Fine-Tuned Model:**
> Hello, thanks for the query. Chest pain for 2 days is not a normal condition. This can be due to infection in the throat or in the chest. You should go for a chest X-ray to be ruled out for any infection, especially if you have a high fever, cough and have been having a sore throat. Hope this helps.

---

## Requirements
```
torch
transformers
datasets
peft
accelerate
trl
rouge-score
bert-score
huggingface-hub
```
