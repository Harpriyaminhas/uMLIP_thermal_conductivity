<!-- <h1 align="center">MatterSim_Thermal_confutivity</h1> -->

<h4 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2405.04967-blue?logo=arxiv&logoColor=white.svg)](https://arxiv.org/abs/2405.04967)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI Downloads](https://static.pepy.tech/badge/mattersim)](https://pepy.tech/projects/mattersim)
</h4>


Unravelling Lone Pair Induced Bonding Effects on Thermal Conductivity in Metal Chalcogenides using Machine Learning Potentials

## Installation

### Prerequisite
* `Python >= 3.10`


### Install from PyPI
> [!TIP]
> While not mandatory, we recommend creating a clean conda environment before installing MatterSim to avoid potential package conflicts. You can create and activate a conda environment with the following commands:
>
> ```bash
> # create the environment
> conda create -n mattersim python=3.10
>
> # activate the environment
> conda activate mattersim
> ```
>

To install MatterSim, use the following command. Please note that downloading the dependencies may take some time:
```bash
pip install mattersim
```

In case you want to install the package with the latest version, you can run the following command:

```bash
pip install git+https://github.com/microsoft/mattersim.git
```

### Install from source code
1. Download the source code of MatterSim and change to the directory

```bash
git clone git@github.com:microsoft/mattersim.git
cd mattersim
```

2. Install MatterSim

To install the package, run the following command under the root of the folder:

```bash
mamba env create -f environment.yaml
mamba activate mattersim
uv pip install -e .
```

## Pre-trained Models

We currently offer two pre-trained **MatterSim-v1** models based on the **M3GNet** architecture in the [pretrained_models](./pretrained_models/) folder:

## To run phonopy and check dinamical stability 
Phonopy folder contains the two python files, mattersim_ph_batch.py and mattersim_phonopy_class.py. Run mattersim_IFC.py to get the phonon results.
The output results are in results_mattersim folder.

## Fine Tune
Data contained in the Fine_tune folder
To fine tune, run finetune_mattersim.py
generated best_model.pth is used to predict the lattice thermal condutivity.  

## Thermal Condutivity
All the results and the python files are placed in the Thermal _condutivity folder
It also contains the CHGNet, MACEE and MatterSim predicted thermal conditivties benchmarked and validated against DFT thermal condutivity results.
```

## Reference
@article{minhas2025mattersim_thermal_condutivity}
      title={Unravelling Lone Pair Induced Bonding Effects on Thermal Conductivity in Metal Chalcogenides using Machine Learning Potentials},
      author={Harpriya Minhas and Rahul Kumar Sharma and Biswarup Pathak},
      year={2025},

We kindly request that users of MatterSim version 1.0.0 cite our preprint available on arXiv:
```
@article{yang2024mattersim,
      title={MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures},
      author={Han Yang and Chenxi Hu and Yichi Zhou and Xixian Liu and Yu Shi and Jielan Li and Guanzhi Li and Zekun Chen and Shuizhou Chen and Claudio Zeni and Matthew Horton and Robert Pinsler and Andrew Fowler and Daniel Zügner and Tian Xie and Jake Smith and Lixin Sun and Qian Wang and Lingyu Kong and Chang Liu and Hongxia Hao and Ziheng Lu},
      year={2024},
      eprint={2405.04967},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2405.04967},
      journal={arXiv preprint arXiv:2405.04967}
}
```
