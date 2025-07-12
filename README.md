# LLM-SRL
This repository contains the code and prompt template for the ACL Findings 2025 paper "LLMs Can Also Do Well! Breaking Barriers in Semantic Role Labeling via Large Language Models", by Xinxin Li*, Huiyao Chen*, Chengjun Liu, Jing Li, Meishan Zhang†, Jun Yu, Min Zhang.

Please cite our paper if it is helpful to your work.

## Introduction
We propose a two-stage SRL framework using large language models (LLMs), consisting of predicate identification and argument labeling. To enhance LLM performance, we introduce two key mechanisms:

(1) Retrieval-Augmented Agent: We augment LLMs with external knowledge from predicate-argument descriptions to improve their understanding of predicate semantics.

(2) Self-Correction: We enable LLMs to evaluate and revise their own outputs to reduce hallucinations.

These mechanisms are integrated into a conversation-based architecture, where reasoning is iterative and LLMs adapt to the SRL task. This approach aligns with traditional BERT-style models while leveraging the strengths of LLMs in reasoning and flexibility.

![framework](./assets/framework.jpg)

## Requirements:
To run our code, please install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```

If you want to accelerate the inference process using vLLM, you can create a new environment that includes the following dependencies:
```
pip install -r vllm_requirements.txt
```

## Data Preprocessing
For replicating results on CPB1.0 (Chinese), CoNLL09 (both Chinese and English), CoNLL12(English) datasets, please follow the steps below.

### CPB1.0
Chinese Proposition Bank 1.0 was developed by the [Linguistic Data Consortium (LDC)](https://catalog.ldc.upenn.edu/LDC2005T23).


### CoNLL-2012
You have to follow the instructions below to get CoNLL-2012 data
[CoNLL-2012](https://conll.cemantix.org/2012/data.html), this would result in a directory called `/path/to/conll-formatted-ontonotes-5.0`.
Run:  
`./scripts/make_conll2012_data.sh /path/to/conll-formatted-ontonotes-5.0`

## Agent Construction


## Conversation Instruction Generation

## Training

## Inference