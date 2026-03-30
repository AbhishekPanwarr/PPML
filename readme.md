[![Fhenix](https://img.shields.io/badge/Fhenix-Off--Chain_FHE-4f46e5)](#)
[![Rust](https://img.shields.io/badge/Rust-PPML_Engine-orange)](#)
[![Python](https://img.shields.io/badge/Python-BlindML-blue)](#)

# PPML

`PPML` is the off-chain privacy-preserving machine learning engine.

It contains only:

- Rust crates for encrypted tensor math, quantization, training, and model export
- the `blindml` Python bridge built with PyO3/Maturin
- local CPU/GPU shell scripts for encrypted training and inference workflows

It does not contain any smart contracts, Hardhat workspace, frontend code, or backend API code.

## Repository layout

```text
PPML/
├── blindml/        # PyO3 Python bridge over the Rust engine
├── ppml_core/      # Rust FHE runtime, tensors, quantization, exporters
├── ppml_train/     # Training/inference binaries
├── scripts/        # Local CPU/GPU helper scripts
├── requirements.txt
└── Cargo.toml
```

## Local setup

```bash
cd /home/abhieren/Drive/Projects/Buildathon/Fhenix/PPML
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
maturin develop --release -m blindml/Cargo.toml
```

## Notes

- The exported model artifact used by the application layer is `model_export.json` at the PPML repo root when training/export completes.
- All blockchain contracts, deployment scripts, frontend ABI usage, and backend/web3 orchestration now live in the private `blindference` repository.
