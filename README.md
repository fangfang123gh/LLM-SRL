# LLM-SRL: Semantic Role Labeling via Large Language Models

> **Paper**: "LLMs Can Also Do Well! Breaking Barriers in Semantic Role Labeling via Large Language Models"  
> **Published**: ACL Findings 2025  
> **Authors**: Xinxin Li*, Huiyao Chen*, Chengjun Liu, Jing Li, Meishan Zhang†, Jun Yu, Min Zhang

## 📖 Project Overview

This project presents a novel two-stage framework for Semantic Role Labeling (SRL) using Large Language Models. By combining **Retrieval-Augmented Agent** and **Self-Correction Mechanism**, we achieve state-of-the-art performance on CPB1.0, CoNLL-2009, and CoNLL-2012 datasets.

### 🎯 Key Innovations

1. **Retrieval-Augmented Agent** - Injects external predicate-argument frame knowledge during predicate identification and argument labeling
2. **Self-Correction Mechanism** - Trains LLMs to identify and iteratively correct inconsistencies in their own SRL outputs
3. **Conversation-Based Two-Stage Architecture** - Stage 1: Predicate identification; Stage 2: Argument labeling

![Framework](https://github.com/fangfang123gh/LLM-SRL/raw/main/assets/framework.jpg)

---

## 🚀 Quick Start

### Environment Setup

**Basic Environment**:
```bash
pip install -r requirements.txt
```

**vLLM Acceleration Environment** (Optional, for faster inference):
```bash
# Create a separate conda environment
conda create -n vllm-env python=3.8
conda activate vllm-env
pip install -r vllm_requirements.txt
```

---

## 📊 Data Preparation

### Supported Datasets

| Dataset | Language | Source |
|---------|----------|--------|
| CPB1.0 | Chinese | [LDC2005T23](https://catalog.ldc.upenn.edu/LDC2005T23) |
| CoNLL-2009 | Chinese & English | [LDC2012T03](https://catalog.ldc.upenn.edu/LDC2012T03) |
| CoNLL-2012 | English | [Official Site](https://conll.cemantix.org/2012/data.html) |

### Data Format

The processed data format is JSON with the following structure:

```json
{
  "text": "sentence text",
  "srl": [
      {
          "pred": "predicate",
          "position": [start_pos, end_pos],
          "arguments": [
              {
                  "value": "argument_value", 
                  "position": [start_pos, end_pos], 
                  "role": "role_label"
              }
          ]
      }
  ],
  "token": ["tokenized", "words"],
  "pos": ["part-of-speech", "tags"],     // Required for CoNLL datasets
  "lemmas": ["lemmatized", "forms"]      // Required for CoNLL datasets
}
```

> **Note**: Position indices use **1-based** numbering (starting from 1)

---

## 🔧 Complete Pipeline

### Step 1: Construct Agent Database

Run the appropriate scripts based on your dataset:

**English Datasets (CoNLL-2012)**:
```bash
python agent_scripts/construct_database_conll12_en.py
python agent_scripts/construct_agent_conll12_en.py
```

**Chinese Datasets (CPB1.0, CoNLL-2009 Chinese)**:
```bash
python agent_scripts/construct_database_zh.py
python gpt_infer.py
python get_chinese_pred_des.py
```

### Step 2: Generate Conversation Instructions

```bash
# Run the instruction generation script for your dataset
python agent_scripts/generate_instructions_[dataset].py
```

### Step 3: Model Training

1. **Edit Training Configuration**:
 ```bash
 vim examples/train_lora/llama3_lora_sft_ds0.yaml
 ```

2. **Start Training**:
 ```bash
 # Adjust the training command based on your configuration
 python train.py --config examples/train_lora/llama3_lora_sft_ds0.yaml
 ```

### Step 4: Model Inference

Choose your inference method:

**Standard Inference**:
```bash
python chat_[dataset].py
```

**vLLM Accelerated Inference**:
```bash
python chat_[dataset]_vllm.py
```

### Step 5: Post-processing and Evaluation

**Post-processing**:
```bash
# For English datasets
python process_rl_en.py

# For Chinese datasets  
python process_rl_zh.py
```

**Evaluation**:
```bash
python evaluate.py
```

---

## 📁 Project Structure

```
LLM-SRL/
├── agent_scripts/          # Agent construction scripts
├── examples/              # Training configuration examples
├── assets/               # Project assets
├── requirements.txt      # Basic dependencies
├── vllm_requirements.txt # vLLM dependencies
├── chat_*.py            # Inference scripts
├── process_rl_*.py      # Post-processing scripts
└── evaluate.py          # Evaluation script
```

---

## 🎯 Usage Guidelines

1. **First-time Users**: Recommend testing the complete pipeline on a small dataset first
2. **Resource Requirements**: Training requires substantial GPU memory; inference can be accelerated with vLLM
3. **Dataset Selection**: Choose appropriate language and dataset based on your task requirements
4. **Parameter Tuning**: Adjust parameters in the training configuration file according to your specific task

---

## 📈 Performance

Our method achieves state-of-the-art results on:
- **CPB1.0** (Chinese Proposition Bank)
- **CoNLL-2009** (Chinese & English)
- **CoNLL-2012** (English)

The two-stage framework with retrieval augmentation and self-correction significantly outperforms previous generative LLM approaches for SRL tasks.

---

## 🔬 Technical Details

### Retrieval-Augmented Agent
- Provides external predicate-argument frame knowledge
- Enhances semantic accuracy during both stages
- Reduces hallucinations in argument identification

### Self-Correction Mechanism
- Iteratively refines LLM outputs
- Identifies and corrects inconsistencies
- Improves overall labeling consistency

### Two-Stage Architecture
- **Stage 1**: Predicate identification with context understanding
- **Stage 2**: Argument labeling with enhanced reasoning
- Both stages benefit from retrieval and self-correction

---

## 📚 Citation

If this work is helpful to your research, please cite:

```bibtex
@article{DBLP:journals/corr/abs-2506-05385,
author    = {Xinxin Li and Huiyao Chen and Chengjun Liu and 
             Jing Li and Meishan Zhang and Jun Yu and Min Zhang},
title     = {LLMs Can Also Do Well! Breaking Barriers in Semantic Role Labeling
             via Large Language Models},
year      = {2025},
url       = {https://doi.org/10.48550/arXiv.2506.05385},
}
```

---

## 🤝 Contributing

- 📧 **Contact**: Please submit issues or contact the authors for questions
- 🌟 **Star Support**: If this project helps you, please give it a star
- 🔄 **Pull Requests**: Contributions and improvements are welcome

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

We thank the developers of the datasets and the open-source community for their valuable contributions to semantic role labeling research.
