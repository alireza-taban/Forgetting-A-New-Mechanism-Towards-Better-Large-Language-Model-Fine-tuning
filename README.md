# FORGETTING: A New Mechanism Towards Better Large Language Model Fine-Tuning

<a href='https://github.com/AliTaheri2002/Forgetting-A-New-Mechanism-Towards-Better-Large-Language-Model-Fine-tuning'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://openreview.net/pdf?id=s36smEoUoX'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a>

[Ali Taheri*](https://alitaheri2002.github.io/), [Alireza Taban*](https://www.linkedin.com/in/alireza-taban-90a460121/), [Qizhou Wang](https://qizhouwang.github.io/homepage/), [Shanshan Ye](https://cassie133ye.github.io/), [Abdolreza Mirzaei](https://people.iut.ac.ir/en/mirzaei), [Tongliang Liu](https://tongliang-liu.github.io/), [Bo Han](https://bhanml.github.io/)

*Equal contribution

Isfahan University of Technology, Hong Kong Baptist University, University of Technology Sydney, Simon Fraser University, The University of Sydney
-----
## Abstract

Supervised fine-tuning (SFT) is crucial for adapting pretrained large language models to specialized tasks, but its effectiveness is heavily dependent on data quality. We propose **FORGETTING**, a novel mechanism that categorizes tokens into positive and negative sets based on influence scores. Rather than discarding low-quality data, our approach actively forgets negative tokens while learning from positive ones. This maintains the full dataset scale while establishing clearer knowledge boundaries, significantly improving model generalization across diverse benchmarks.


## Method Overview

Traditional SFT approaches face a fundamental trade-off: either train on all data uniformly (risking overfitting to noise) or discard low-quality samples (reducing dataset scale). Our **FORGETTING** mechanism resolves this by:

1. **Reference Model Training**: Fine-tune the base model on a separate subset to establish a baseline
2. **Token Quality Assessment**: Compute cross-model influence scores to quantify each token's informativeness
3. **Adaptive Partitioning**: Split tokens into positive (top ρ%) and negative sets based on quality scores
4. **Dual-Objective Training**: Maximize likelihood of positive tokens while minimizing likelihood of negative tokens using an adaptive balancing coefficient λ(step)

This approach preserves dataset scale while leveraging negative samples as learning signals to define clearer knowledge boundaries.

## Quick Start

### Training

Execute the complete pipeline (preprocessing, reference model training, influence calculation, and final training):
```bash
bash run_pipeline.sh
```

### Evaluation

Evaluate your trained model on our benchmark suite:
```bash
bash run_evaluation.sh <model_size>
```

## Experimental Results

Our forgetting mechanism consistently outperforms standard SFT across multiple model architectures and scales:

| Base Model | Standard SFT | Forgetting (Ours) | Absolute Gain |
|------------|--------------|-------------------|---------------|
| LLaMA-3.2-1B | 30.37% | **34.86%** | +4.49% |
| LLaMA-3.2-3B | 46.90% | **52.18%** | +5.28% |
| LLaMA-3.1-8B | 51.02% | **59.27%** | +8.25% |
| LLaMA-2-13B | 39.57% | **46.28%** | +6.71% |
| Qwen2.5-3B | 42.52% | **59.01%** | +16.49% |
| GPT-Neo-2.7B | 23.78% | **28.15%** | +4.37% |

*Average performance across TruthfulQA, BoolQ, LogiQA, TydiQA, and ASDiv benchmarks.*

## Citation
```bibtex
@article{taheri2025forgetting,
  title={Forgetting: A New Mechanism Towards Better Large Language Model Fine-tuning},
  author={Taheri, Ali and Taban, Alireza and Ye, Shanshan and Mirzaei, Abdolreza and Liu, Tongliang and Han, Bo},
  journal={arXiv preprint arXiv:2508.04329},
  year={2025}
}
```
