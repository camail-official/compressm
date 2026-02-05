<p align="center">
  <img src="assets/matryoshka_compressm.png" width="600" alt="CompreSSM">
</p>

<h1 align="center">CompreSSM</h1>

<p align="center">
  <b>The Curious Case of In-Training Compression of State Space Models</b><br>
  <a href="https://arxiv.org/abs/2510.02823">Paper</a> • ICLR 2026 • <a href="https://github.com/phnazari/compreSSMamba">Mamba Experiments</a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/google/jax"><img src="https://img.shields.io/badge/JAX-0.4+-green.svg" alt="JAX"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

This is the main repository accompanying the ICLR 2026 paper [The Curious Case of In-Training Compression of State Space Models](https://arxiv.org/abs/2510.02823). It contains the LRU experiments. The Mamba experiments are available in the [CompreSSMamba](https://github.com/phnazari/compreSSMamba) repository.

---

State Space Models (SSMs) offer parallelizable training and fast inference for long sequence modeling. At their core are recurrent dynamical systems with update costs scaling with state dimension. **CompreSSM** applies balanced truncation—a classical control-theoretic technique—*during training* to identify and remove low-influence states based on their Hankel singular values. Models that begin large and shrink during training achieve computational efficiency while maintaining higher performance than models trained directly at smaller dimension.

---

## Installation

```bash
git clone https://github.com/camail-official/compressm.git
cd compressm
conda env create -f environment.yaml
conda activate compressm
```


## Data Preparation

The datasetsMNIST and CIFAR will auto-download into the `data/` directory. [LRA](#long-range-arena-lra) must be manually downloaded from the [GitHub page](https://github.com/google-research/long-range-arena).
These datasets should be organized as follows:
```
path/to/data/
  pathfinder/
    pathfinder32/
    pathfinder64/
    pathfinder128/
    pathfinder256/
  aan/
  listops/
```

## Quick Start

```bash
# Train baseline (no compression)
python scripts/train.py --config configs/paper/smnist_baseline.yaml --seed 42

# Train with τ=0.01 compression (discard 1% Hankel energy)
python scripts/train.py --config configs/paper/smnist_tau0.01.yaml --seed 42
```

## Paper Reproduction

Config files for all experiments are in `configs/paper/`:

```bash
# sMNIST (Table 2) - 10 seeds
python scripts/reproduce.py configs/paper/smnist_baseline.yaml --seeds 8 42 123 456 789 101 202 303 404 505 --gpu 0
python scripts/reproduce.py configs/paper/smnist_tau0.01.yaml --seeds 8 42 123 456 789 101 202 303 404 505 --gpu 0
python scripts/reproduce.py configs/paper/smnist_tau0.02.yaml --seeds 8 42 123 456 789 101 202 303 404 505 --gpu 0
python scripts/reproduce.py configs/paper/smnist_tau0.04.yaml --seeds 8 42 123 456 789 101 202 303 404 505 --gpu 0

# sCIFAR (Table 3) - 5 seeds
python scripts/reproduce.py configs/paper/scifar_baseline.yaml --seeds 8 42 123 456 789 --gpu 0
python scripts/reproduce.py configs/paper/scifar_tau0.05.yaml --seeds 8 42 123 456 789 --gpu 0
python scripts/reproduce.py configs/paper/scifar_tau0.10.yaml --seeds 8 42 123 456 789 --gpu 0
python scripts/reproduce.py configs/paper/scifar_tau0.15.yaml --seeds 8 42 123 456 789 --gpu 0

# Aggregate results
python scripts/analyse_results.py outputs/paper/ --output results/
```

### Expected Results

| Config | Table | τ | Accuracy | Final Dim |
|--------|-------|---|----------|-----------|
| `smnist_baseline` | 2 | 0% | ~97.3% | 256 |
| `smnist_tau0.01` | 2 | 1% | ~96.9% | ~47 |
| `smnist_tau0.02` | 2 | 2% | ~96.9% | ~28 |
| `smnist_tau0.04` | 2 | 4% | ~95.9% | ~13 |
| `scifar_baseline` | 3 | 0% | ~86.5% | 2304 |
| `scifar_tau0.05` | 3 | 5% | ~85.8% | ~161 |
| `scifar_tau0.10` | 3 | 10% | ~85.7% | ~93 |
| `scifar_tau0.15` | 3 | 15% | ~84.4% | ~57 |

## Code Structure

```
compressm/
├── models/lru.py                  # LRU model with reduction
├── reduction/
│   ├── hsv.py                     # Hankel singular value computation
│   └── balanced_truncation.py     # Balanced truncation algorithm
├── training/trainer.py            # Training loop with in-training compression
└── data/datasets.py               # sMNIST, sCIFAR loaders

configs/paper/                     # Paper reproduction configs
scripts/
├── train.py                       # Training CLI
├── reproduce.py                   # Multi-seed reproduction
└── analyse_results.py             # Results aggregation
```

## Citation

```bibtex
@misc{chahine2026curiouscaseintrainingcompression,
      title={The Curious Case of In-Training Compression of State Space Models}, 
      author={Makram Chahine and Philipp Nazari and Daniela Rus and T. Konstantin Rusch},
      year={2026},
      eprint={2510.02823},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.02823}, 
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
