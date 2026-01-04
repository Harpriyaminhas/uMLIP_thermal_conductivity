<!-- <h1 align="center">MatterSim_Thermal_confutivity</h1> -->

Unravelling Lone Pair Induced Bonding Effects on Thermal Conductivity in Metal Chalcogenides using Machine Learning Potentials

## Installation

### Prerequisite
* `Python >= 3.10`


### Install from PyPI
> [!TIP]
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

## To run phonopy and check dinamical stability 
Phonopy folder contains the two python files, mattersim_ph_batch.py and mattersim_phonopy_class.py. Run mattersim_IFC.py to get the phonon results.
The output results are in results_mattersim folder.

## Fine Tune
Data contained in the Fine_tune folder
To fine tune, run finetune_mattersim.py
generated best_model.pth is used to predict the lattice thermal condutivity.  

## Thermal Condutivity
All the results and the python files are placed in the Thermal _condutivity folder
It also contains the CHGNet, MACE and MatterSim predicted thermal conditivties benchmarked and validated against DFT thermal condutivity results.
Similarly install CHGNet and MACE to predict the Dynamical Stability
```

## Reference
@article{minhas2025mattersim_thermal_condutivity}
  title   = {Unravelling Lone Pair--Induced Bonding Effects on Thermal Conductivity in Metal Chalcogenides Using Machine Learning Potentials},
  author  = {Minhas, Harpriya and Sharma, Rahul Kumar and Pathak, Biswarup},
  journal = {Journal of Materials Chemistry A},
  year    = {2025},
  volume  = {13},
  doi     = {10.1039/D5TA08916F},
  publisher = {Royal Society of Chemistry}
}

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
