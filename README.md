# CROWN-Reach

CROWN-Reach is a new open-source tool for reachability analysis of neural network control systems developed at UIUC. It aims to strengthen and extend the successful [α-β-CROWN neural network verifier](https://github.com/Verified-Intelligence/alpha-beta-CROWN) to the setting of neural network controller verification. 

CROWN-Reach consists of four main components: bound-propagation for efficient analysis of neural network controllers, Taylor model for plant analysis, branch-and-bound to refine the reachable set, and a sampling-based falsifier. For the analysis of neural network controllers, we use linear relaxation based perturbation analysis (LiRPA) methods such as [CROWN](https://arxiv.org/pdf/1811.00866) and [α-CROWN](https://arxiv.org/pdf/2011.13824) with extensions to cooperate with Taylor Model flowpipe computation.

Our tool is based on the [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) library, which can automatically compute linear functional over-approximations for neural networks with various activation functions, including ReLU, tanh, and sigmoid, as well as neural networks with general architectures (e.g., residual blocks and custom operators). We use the [Flow*](https://github.com/chenxin415/flowstar) library for analyzing the plant with continuous dynamics using Taylor models, and these Taylor models are symbolically combined with the linear bounds from CROWN to form the reachable set of the entire system. 
The branch-and-bound refinement process splits the input state space and utilizes parallelization (including both GPU and CPU) to achieve quick and precise analysis. The bound propagation process can be accelerated on GPUs and can scale to very large networks, while the computation of Taylor models is executed on CPUs using multi-threading. A paper describing the algorithm details of CROWN-Reach is currently being prepared.


## Installation and Setup
Clone this repository:
```bash
git clone https://github.com/xiangruzh/CROWN-Reach.git
```

Install dependencies:
```bash
# libraries for Flow*
sudo apt-get install m4 libgmp3-dev libmpfr-dev libmpfr-doc libgsl-dev gsl-bin bison flex
# libraries for rpc
sudo apt-get install libjsoncpp-dev libcurl4-openssl-dev libjsonrpccpp-dev libjsonrpccpp-tools
# library for multi-thread
sudo apt-get install libboost-all-dev
```

Clone [Flow*](https://github.com/chenxin415/flowstar) to ```./CROWN-Reach/flowstar``` and compile.
```bash
git clone https://github.com/chenxin415/flowstar.git ./CROWN-Reach/flowstar
cd ./CROWN-Reach/flowstar/flowstar-toolbox
make
```

Setup the Conda environment:
```bash
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name CROWN_Reach
# install all dependents into the CROWN_Reach environment
conda env create -f environment.yaml --name CROWN_Reach
# activate the environment
conda activate CROWN_Reach
```

## Instructions for running the ARCH-COMP benchmarks
You run our tool on the ARCH-COMP benchmarks with
```
./submit.sh
```
The verifier will run using Docker, and the results will be saved at ```./results/results.csv```.

## Developers and Copyright

CROWN-Reach is released under the BSD 3-Clause License. It is developed by Xiangru Zhong (xiangruzh0915@gmail.com) and Yuhao Jia (yuhaojia98@g.ucla.edu), supervised by Prof. [Huan Zhang](https://huan-zhang.com/) at UIUC.
