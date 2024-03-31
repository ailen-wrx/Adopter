# Adopter: Automated Deep Learning Optimization via DSL-based Source Code Transformation

Official implementation for our paper "Automated Deep Learning Optimization via DSL-based Source Code Transformation", ISSTA 2024.

## Overview

```
Adopter
 ├── dsl.py              Domain-specific langiage (§3.1)
 |                       and model structure abstraction (§3.2)
 ├── pattern_matcher.py  Pattern matching (§3.3)
 ├── refactor.py         Synthesis-based code transformation (§3.4)
 ├── ...
 ├── models/             Hugging Face models used for evaluation
 ├── benchmarks/         Benchmarking Optimization Rules (§4.4.4)
 └── results/            Generated patches and statistic results
      ├── stat_adopter.tsv    Adopter performance    (§4.4.1)
      ├── stat_comby.tsv      Baseline performance
      ├── stat_adopter.tsv    Ablation study         (§4.4.3)
      └── time.tsv            Runtime overhead       (§4.4.5)

```

## Deep Learning Optimization

Adopter includes 9 deep learning model optimization rules.

| Name                | Description                                           | Source                                                    |
|---------------------|-------------------------------------------------------|-----------------------------------------------------------|
| `BertSelfAttention` | `BertSelfAttention` with xformers’ attention ops      | [xformers](https://github.com/facebookresearch/xformers)   |
| `T5Attention`       | `T5Attention` with xformers’ attention ops            | [xformers](https://github.com/facebookresearch/xformers)   |
| `GPT2Attention`     | `GPT2Attention` with xformers’ attention ops          | [xformers](https://github.com/facebookresearch/xformers)   |
| `softmax`           | A drop-in replacement to `torch.nn.softmax`           | [triton](https://github.com/openai/triton)                 |
| `Dropout_LayerNorm` | Fusing `Dropout` and `LayerNorm`                      | [epoi](https://github.com/comaniac/epoi)                   |
| `biased_GeLU`       | Fusing biased `Linear` and `GeLU` activation          | [epoi](https://github.com/comaniac/epoi)                   |
| `Conv_BatchNorm`    | Fusing `Conv2d` and `BatchNorm2d`                     | [PyTorch](https://github.com/pytorch/pytorch)              |
| `Linear_BatchNorm`  | Fusing `Linear` and `BatchNorm1d`                     | [PyTorch](https://github.com/pytorch/pytorch)              |
| `fused_QKV`         | Fusing three `Linear` layers as `q`, `k`, and `v` in encoder | [slapo](https://github.com/awslabs/slapo)              |

## Requirements
 -  Operating System: Ubuntu 20.04
 -  Python Version: 3.9.10
 -  PyTorch Version: 2.0.1
 -  CUDA Version: 11.7
 -  Python dependencies:
     -   python_graphs
     -   ast
     -   gast
     -   astunparse
     -   importlib
     -   inspect


## Run Adopter
```
git submodule update --init     # initialize submodules
python3 main.py all 
python3 main.py all ablation    # ablation study
```

## Checkout Evaluation Result
```
cd results
python3 compare.py
```
Check out the `.tsv` files under the directory `results/`.

## Optimization Benchmarks
Please follow `benchmarks/benchmarks.ipynb` to check out the results.
