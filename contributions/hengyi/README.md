# Hengyi Yin — GPU Calibration and Request-Triage Prototype

**Contributor:** [Hengyi Yin (@hennyi-yin)](https://github.com/hennyi-yin)

Start with **[REPORT.md](REPORT.md)**: my work summary, measured results, evidence limits, and proposed integration tasks.

This folder is self-contained. Run commands **from this directory**, not the repository root. Python 3.11+ is required.

## Review without a GPU

```powershell
python -m pip install -r requirements-rl.txt
python -m unittest discover -s tests -v
python verify_evidence.py
```

Retained evidence covers 102 actual GPU timing runs, the checked real queue replay, and compact calibrated-simulation summaries. No weights, virtual environments, full datasets, or bulk simulation outputs are committed.

## Reproduce measured-profile RL

```powershell
python prepare_trace.py
python train_dqn.py --profile profiles/rtx5070_qwen05b.json
python evaluate_dqn.py --profile profiles/rtx5070_qwen05b.json
```

Trace preparation downloads the pinned public BurstGPT CSV (about 51 MB), filters requests, and selects chronological windows. Downloaded/processed data and regenerated training/evaluation outputs are ignored. Policy workload statistics use training windows only.

This is simulation using measured service times, **not DQN executing in a real GPU queue**.

## Reproduce the GPU experiment

Recorded backend: Windows, RTX 5070, FP16, PyTorch 2.9.0+cu130, Transformers 4.57.1, SDPA, batch=1. A different backend requires a new profile.

```powershell
python -m venv .venv-gpu
.venv-gpu/Scripts/python.exe -m pip install -r requirements-rl.txt -r requirements-gpu.txt
.venv-gpu/Scripts/python.exe fetch_gpu_model.py
.venv-gpu/Scripts/python.exe gpu_benchmark.py
.venv-gpu/Scripts/python.exe gpu_benchmark.py --prompts 32 128 512 2048 --outputs 1 2 8 32 --output results/gpu/short
.venv-gpu/Scripts/python.exe calibrate_gpu.py
.venv-gpu/Scripts/python.exe real_gpu_triage.py --seed 9000 --output results/gpu/reproduction_queue
```

Timing defaults overwrite canonical measurements. Preserve the retained evidence first, or use `--output` and pass the new CSV paths to `calibrate_gpu.py --measurements ... --output ...`. Keep each CSV beside its matching `environment.json`. The replay command above writes a new ignored output directory. Dependencies and model weights require several GB; downloaded weights are pinned and hash-checked.

To regenerate the figure:

```powershell
python -m pip install -r requirements-plots.txt
python plot_results.py
```

## File map

| Files | Purpose |
|---|---|
| `triage_sim.py`, `rl_env.py`, `service_profile.py` | Queue accounting, visible observations, bounded Defer, timing model |
| `dqn.py`, `train_dqn.py`, `evaluate_dqn.py` | Double-DQN and validation-only checkpoint selection |
| `prepare_trace.py`, `workloads.py` | Pinned BurstGPT acquisition and chronological replay |
| `fetch_gpu_model.py`, `gpu_benchmark.py` | Pinned model and actual CUDA timing |
| `calibrate_gpu.py`, `profiles/` | Timing fit and untouched-length validation |
| `real_gpu_triage.py` | Actual wall-clock FCFS/static queue replay |
| `results/gpu/` | Minimal raw hardware evidence and meeting figure |
| `results/gpu_evaluation/`, `results/gpu_models/` | Compact simulation summaries/configuration, without checkpoints |
| `tests/`, `verify_evidence.py` | Behavioral tests and evidence checks |

## Attribution

The download recipe uses [BurstGPT](https://github.com/HPMLL/BurstGPT), CC-BY-4.0, and saves its license/provenance locally. Model provenance is in `results/gpu/model_source.json`; [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) is Apache-2.0. No third-party dataset or model weights are redistributed here.
