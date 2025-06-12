# Leveraging Small Language Models for BPMN Generation

This repository contains the implementation of our research on using Small Language Models (SLMs) to automatically generate Business Process Model and Notation (BPMN) diagrams from natural language descriptions.

> ⚠️ **Work in Progress**: Please note that some parts of the codebase are not yet fully optimized and the work is still ongoing.

## 🎯 Overview

We investigate whether Small Language Models can accurately translate natural-language process descriptions into BPMN models through a three-stage adaptation process:

1. **Few-shot prompting** - Conditioning outputs on representative examples
2. **Supervised fine-tuning with LoRA** - Domain-specific model specialization
3. **Reinforcement learning via GRPO** - Improving structural correctness

Our 3-billion-parameter LLaMA 3.2 model achieves:
- **0.80 structural similarity** and **0.84 structural correctness** after LoRA fine-tuning
- **0.89 structural correctness** after GRPO

## 🚀 Key Features

- **Privacy-preserving** - Local training and inference, no external API calls
- **Cost-efficient** - Runs on commodity hardware (NVIDIA Tesla V100)
- **Open-weight** - Based on LLaMA 3.2-3B model
- **Compact representations** - Two intermediate formats (MER and JSON)
- **Multiple adaptation strategies** - Few-shot, LoRA, and GRPO techniques


## 📊 Dataset

We use a combination of datasets:
- **Training**: 2,250 BPMN-text pairs (90% train, 10% validation)
  - 1,125 examples from [MaD dataset](https://drive.google.com/drive/u/0/folders/1n0K9BmiDsXYCqB796MVebYBgWX2ruZpW)
  - 1,125 high-quality models from [SAP-SAM dataset](https://github.com/signavio/sap-sam)
- **Evaluation**: [Zenodo dataset](https://zenodo.org/records/7783492) with 212 expert-reviewed BPMN models across 24 textual descriptions

Examples of how the training dataset looks like are available in `/dataset/` for each format and fine-tuning phase.


## 📦 Environment Setup

Before starting, make sure you have the following installed:

- [Python](https://www.python.org/)
- [Conda](https://docs.conda.io/en/latest/)

Then, create a conda environment from the provided `env.yml` file:

```bash
conda env create -f env.yml
```

> ⚠️ Note: The environment creation process can be a little slow. Also, for Windows user, there might be some issues during setup, which still need to be addressed. 

---

## 📁 Project Structure

```
slm-for-bpmn/
├── src/
│   ├── utils/                      # Collection of various utilities used in the project
│   ├── compute_metrics.py          # Compute structural similarity and structural correctness
│   ├── converter.py                # Convert XML/DOT file in JSON/MER
│   ├── create_dataset.py           # Generate training and validation dataset for LoRA and GRPO fine-tuning (in both MER and JSON format)
|   ├── supervised_finetuning.py    # LoRA fine-tuning script
|   ├── grpo_finetuning.py          # GRPO fine-tuning script
│   └── zenodo_inference.py         # Run inference on Zenodo 24 textual descriptions
├── dataset/
│   ├── GRPO/train_example.json     # Examples taken from the training set for GRPO fine-tuning look like (using JSON format)
│   ├── JSON/train_example.json     # Examples taken from the training set for LoRA fine-tuning and JSON format
│   └── MER/train_example.json      # Examples taken from the training set for LoRA fine-tuning and MER format            
└── prompts/    
    ├── JSON/                           # system prompts for JSON format
    │   ├── 4shots_prompt.txt               # system prompt to run inference on plain model using few-shot prompting
    │   ├── SFT_prompt.txt                  # system prompt used for LoRA finetuning
    │   └── GRPO_prompt.txt                 # system prompt used for GRPO refinement
    ├── MER/                            # system prompts for MER format
    │   ├── 4shots_prompt.txt               # system prompt to run inference on plain model using few-shot prompting
    │   └── SFT_prompt.txt                  # system prompt used for LoRA finetuning
    ├── GPT4o_prompt.txt                # prompt used to generate BPMN models with GPT-4o
    └── GPT4o_generate_descriptions.txt # prompt used to generate missing textual descriptions for SAP-SAM data 
```