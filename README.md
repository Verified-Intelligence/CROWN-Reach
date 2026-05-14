# CROWN-Reach

CROWN-Reach is a neural-network control-system reachability tool built around:

- CROWN-based neural bound propagation in Python.
- Flow* Taylor-model plant analysis in C++.
- A branch-and-bound refinement loop in C++.
- A falsification front-end (`src/attack.py`) before full verification.

The repository has two execution modes:

- Local research mode in `src` (single benchmark run from YAML config).
- Batch competition mode in `submit` (Dockerized ARCH-COMP benchmark sweep).

## System Dependencies

Install required native libraries:

```bash
sudo apt-get update
sudo apt-get install -y \
	m4 libgmp3-dev libmpfr-dev libmpfr-doc libgsl-dev gsl-bin bison flex libglpk-dev \
	libjsoncpp-dev libcurl4-openssl-dev libjsonrpccpp-dev libjsonrpccpp-tools \
	libboost-all-dev \
	libyaml-cpp-dev \
	build-essential cmake git python3 python3-pip
```

## Quick Setup (Recommended)

```bash
git clone https://github.com/Verified-Intelligence/CROWN-Reach.git
cd CROWN-Reach

# Optional: create/activate your conda environment first.
conda env create -f environment.yaml -n CROWN_Reach
conda activate CROWN_Reach

# Clone external repos and install Python dependencies.
./scripts/setup_external_deps.sh
```

What this does:

- Clones Flow* to `./flowstar`.
- Clones auto_LiRPA to `./external/auto_LiRPA`.
- Installs Python requirements from `requirements.txt`.
- Installs auto_LiRPA in editable mode.

## Local Run (src)

Build the C++ executable:

```bash
cd src
make
```

Run a benchmark config:

```bash
python run.py configs/tora_sigmoid.yaml
```

Main local files:

- `src/run.py`: orchestration (attack -> CROWN RPC server -> C++ verifier).
- `src/crown.py`: RPC server producing CROWN linear bounds from ONNX models.
- `src/CrownSettings.cpp` and `src/CrownReach.cpp`: C++ verification core.
- `src/configs/*.yaml`: benchmark-specific settings.

## Docker Batch Run (submit)

Build and run all ARCH-COMP benchmark instances:

```bash
cd submit
./submit.sh run1
```

Results are copied to:

- `submit/results/results.csv`

## Supported Benchmarks

The batch runner in `submit/run.py` covers:

- ACC
- Airplane
- Attitude Control
- Single Pendulum
- TORA (standard, sigmoid, relu+tanh)
- Unicycle
- CartPole
- VCAS
- Double Pendulum (more-robust, less-robust)
- NAV (robust, standard)
- Quadrotor

