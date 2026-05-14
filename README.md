# CROWN-Reach

CROWN-Reach is a new open-source tool for reachability analysis of neural network control systems developed at UIUC. It aims to strengthen and extend the successful [alpha-beta-CROWN neural network verifier](https://github.com/Verified-Intelligence/alpha-beta-CROWN) to the setting of neural network controller verification. CROWN-Reach consists of four main components: bound-propagation for efficient analysis of neural network controllers, Taylor model for plant analysis, branch-and-bound to refine the reachable set, and a sampling-based falsifier. For the analysis of neural network controllers, we use linear relaxation based perturbation analysis (LiRPA) methods such as [CROWN](https://arxiv.org/pdf/1811.00866) and [alpha-CROWN](https://arxiv.org/pdf/2011.13824) with extensions to cooperate with Taylor Model flowpipe computation.

Our tool is based on the [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) library, which can automatically compute linear functional over-approximations for neural networks with various activation functions, including ReLU, tanh, and sigmoid, as well as neural networks with general architectures (e.g., residual blocks and custom operators). We use the [Flow*](https://github.com/chenxin415/flowstar) library for analyzing the plant with continuous dynamics using Taylor models, and these Taylor models are symbolically combined with the linear bounds from CROWN to form the reachable set of the entire system. The branch-and-bound refinement process splits the input state space and utilizes parallelization (including both GPU and CPU) to achieve quick and precise analysis. The bound propagation process can be accelerated on GPUs and can scale to very large networks, while the computation of Taylor models is executed on CPUs using multi-threading. A paper describing the algorithm details of CROWN-Reach is currently being prepared.

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

