# LLM‑SRL

This repository contains the code and prompt templates for the ACL Findings 2025 paper **“LLMs Can Also Do Well! Breaking Barriers in Semantic Role Labeling via Large Language Models”** by Xinxin Li\*, Huiyao Chen\*, Chengjun Liu, Jing Li, Meishan Zhang†, Jun Yu, Min Zhang.

> **Abstract (summary):**
> Generative LLMs have historically underperformed compared to encoder–decoder models for SRL. We propose a two-stage framework combining **(1)** Retrieval‑Augmented Agent (to inject argument knowledge) and **(2)** Self‑Correction (to refine outputs iteratively), achieving state-of-the-art performance on CPB1.0, CoNLL‑2009, and CoNLL‑2012.
![framework](./assets/framework.jpg)
---

## 🚀 Key Contributions

1. **Retrieval‑Augmented Agent**
   – During predicate identification and argument labeling, we provide LLMs with external predicate–argument frame knowledge to enhance semantic accuracy.

2. **Self‑Correction Mechanism**
   – LLMs are trained to identify inconsistencies in their own SRL output and iteratively correct them, mitigating hallucinations and improving consistency.

3. **Conversation‑Based Two‑Stage Architecture**
   – First stage: predicate identification; second stage: argument labeling; both bolster LLM reasoning via retrieval and self-correction.

---

## 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

To accelerate inference using vLLM, create a separate environment and install:

```bash
pip install -r vllm_requirements.txt
```

---

## 📊 Data Preprocessing

To replicate our experiments on CPB1.0 (Chinese), CoNLL-2009 (Chinese & English), and CoNLL-2012 (English), follow the instructions below.

### 📌 CPB1.0

Chinese Proposition Bank 1.0 (LDC2005T23) is developed by [Linguistic Data Consortium (LDC)](https://catalog.ldc.upenn.edu/LDC2005T23).

Processed data format:

```json
{
    "text": str,
    "srl": [
        {
            "pred": str,
            "position": [start, end],
            "arguments": [
                {"value": str, "position": [start, end], "role": str}
            ]
        }
    ],
    "token": []
}
```

Note: `start` is the **1-based index** into the `token` list.

### 📌 CoNLL-2009 & CoNLL-2012

* CoNLL-2009 dataset is from [LDC2012T03](https://catalog.ldc.upenn.edu/LDC2012T03).
* For CoNLL-2012, follow instructions from [deep\_srl GitHub](https://github.com/luheng/deep_srl) and [CoNLL official site](https://conll.cemantix.org/2012/data.html). 
* The dev dataset shoud run
```
python sample_dev.py
```

Processed data format:

```json
{
    "text": str,
    "srl": [...],
    "token": [],
    "pos": [],
    "lemmas": []
}
```

Note: `start` uses **1-based token indexing**.

---

## 🔧 Agent Construction

Run the corresponding script in `./agent_scripts` for each dataset.

This version separates **predicate-level descriptions** and **frame-level role explanations**.

* For **English datasets** (e.g., CoNLL-2012):

  ```bash
  python agent_scripts/construct_database_conll12_en.py
  python agent_scripts/construct_agent_conll12_en.py
  ```

* For **Chinese datasets**:

  ```bash
  python agent_scripts/construct_database_zh.py
  python gpt_infer.py
  python get_chinese_pred_des.py
  ```

---

## ✏️ Conversation Instruction Generation

Generate dataset-specific input instructions by running the appropriate script in the `agent_scripts/` directory.

These files serve as the conversation-style inputs to LLMs during inference.

---

## 🧪 Training

Edit the training configuration in:

```yaml
examples/train_lora/llama3_lora_sft_ds0.yaml
```

Then run training using:

```bash
bash train.sh
```

---

## 🤖 Inference

Run the appropriate `chat_*.py` script for each dataset.
You can choose between:

* **Standard HuggingFace-based inference**, or
* **vLLM-accelerated inference** (for faster throughput)

---

## 📈 Evaluation

After inference, run the following for post-processing:

```bash
python process_rl_en.py      # For English
python process_rl_zh.py      # For Chinese
```

Then evaluate using:

```bash
python metric.py
```

---
## 📬 Citation

If this work is helpful to your research, please cite:

```bibtex
@article{DBLP:journals/corr/abs-2506-05385,
  author       = {Xinxin Li and
                  Huiyao Chen and
                  Chengjun Liu and
                  Jing Li and
                  Meishan Zhang and
                  Jun Yu and
                  Min Zhang},
  title        = {LLMs Can Also Do Well! Breaking Barriers in Semantic Role Labeling
                  via Large Language Models},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2506.05385},
}
```

---

