# GNN_MIP_CAMD

This repository is the official implementation of the paper ["Optimizing over trained GNNs via symmetry breaking"](https://arxiv.org/abs/2305.09420). The paper has been accepted into NeurIPS 2023. Until the conference, please cite this preprint:

- Shiqiang Zhang, Juan S Campos, Christian Feldmann, David Walz, Frederik Sandfort, Miriam Mathea, Calvin Tsay, Ruth Misener. "Optimizing over trained GNNs via symmetry breaking." arXiv preprint arXiv:2305.09420 (2023).

The BibTeX reference is:

    @article{zhang2023optimizing,
    title={Optimizing over trained GNNs via symmetry breaking},
    author={Zhang, Shiqiang and Campos, Juan S and Feldmann, Christian and Walz, David and Sandfort, Frederik and Mathea, Miriam and Tsay, Calvin and Misener, Ruth},
    journal={arXiv preprint arXiv:2305.09420},
    year={2023}
}

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

A license is needed to use *Gurobi*. Please follow the instructions to obtain a [free academic license](https://www.gurobi.com/academia/academic-program-and-licenses/). 

## Data preparation

Two datasets (QM7 and QM9) used in this paper are avaiable in package [*chemprop*](https://chemprop.readthedocs.io/en/latest/tutorial.html#data) (available under a MIT license in ``\data\LICENSE.txt``). We already download and uncompress them here. 

To preprocess the raw datasets, run this command (using QM7 as an example):

```
python QM7_dataset_generation.py
```


## Training and transformation of GNNs

To train a GNN and then transform it into ONNX format, run this command:
```
python training_and_transformation.py $dataset $seed_gnn
```
where \$dataset is the name of dataset (QM7 or QM9), \$seed_gnn is the random seed for traing GNN.


## Count feasible solutions

To count the number of all feasible solutions given N, run this command:
```
python count_feasible.py $dataset $N $break_symmetry
```
where \$dataset is the name of dataset (QM7 or QM9), \$N is the number of atoms, \$break_symmetry controls the level of breaking symmetry.


## Optimization

To solve the optimization problem, run this command:
```
python optimality.py $dataset $N $formulation_type $break_symmetry $seed_gnn $seed_gurobi
```
where \$dataset is the name of dataset (QM7 or QM9), \$N is the number of atoms (chosen from {4,5,6,7,8}), \$formulation_type denotes the type of formulation (0 for bi-linear, 1 for big-M), \$break_symmetry is binary and indices adding symmetry-breaking constraints, \$seed_gnn is the random seed for traing GNN, and \$seed_gurobi is the random seed of Gurobi.

## MIP formulations for CAMD

Constraints (C1) ~ (C25) in the paper correspond to line 88 ~ 245 in ``\optimality.py``. 

The bounds for both datasets are specified in line 35 ~ 52 in ``\optimality.py``. 

The symmetry-breaking constraints (C26) and (C27) are implemented in line 248 ~ 273 in ``\optimality.py``. 


## MIP formulations for GNNs

The implementation of MIP formulation for GNNs is based on open-source package [OMLT](https://github.com/cog-imperial/OMLT) (available under a BSD license in ``\omlt\LICENSE.rst``). We downloaded the source code of OMLT at 8/2/2023, which was still the newest version when we submitted the paper. 

Two formulations ``full_space_gnn_layer_bilinear`` and ``full_space_gnn_layer_bigm`` are added in ``\omlt\neuralnet\layers\full_space.py`` (line 38 ~ 169).

To call one of these formulations, ``\omlt\neuralnet\nn_formulation.py`` is modified (line 185 ~ 190).

# Contributors
Shiqiang Zhang. Funded by an Imperial College Hans Rausing PhD Scholarship.
