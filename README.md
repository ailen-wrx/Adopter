# Adopter: Automated Deep Learning Optimization via DSL-based Source Code Transformation

Ruixin Wang, Minghai Lu, Cody Hao Yu, Yi-Hsiang Lai, Tianyi Zhang

ISSTA'24: Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis

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

## Run Adopter
```
python3 main.py all
python3 main.py all ablation
```

## Checkout Evaluation Result
```
cd results
python3 compare.py
```

## Optimization Benchmarks
Please check out `benchmarks/benchmarks.ipynb`.