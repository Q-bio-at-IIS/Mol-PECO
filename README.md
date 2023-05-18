# Mol-PECO
This is the workspace for odor chemical recognition by laplacian graph neural network. Mol-PECO (**Mol**ecular Representation by **P**ositional **E**ncoding of **Co**ulomb Matrix) is the deep learning framework of extacting molecular representation for odor perception prediction. Mol-PECO's strucutre is as follows:
<div align=center>
<img src="https://github.com/Q-bio-at-IIS/Mol-PECO/blob/main/Mol-PECO.png" width="90%" height="90%">
</div>

### Text
[Overleaf]: https://www.overleaf.com/project/63535977cb446c410c09ee7c
- [Overleaf document][Overleaf]

### Dataset
#### Data source
All comes from pyrfume-data (https://github.com/pyrfume/pyrfume-data). After data cleaning, there exists ```8503``` molecule-odor descriptors pairs, which is 1.5 folds higher than Google's (https://arxiv.org/abs/1910.10685) with ```~5000``` pairs. The folder of ```pyrfume_sois_canon``` mainly includes the preprocessed datasets with:
- Adjacent matrix
- Coulomb matrix
- Atom array
- Eigenvectors and eigenvalues of adjacent matrix
- Eigenvectors and eigenvalues of Coulomb matrix

#### Data split
The dataset was split by second order iterative stratification, same as Google's split method (https://arxiv.org/abs/1910.10685), to handle the imbalanced distribution of odor descriptors with train/val/test of  8/1/1.

### Requirements
- python-3.7+
- deepchem
- matplotlib
- openpyxl
- pyTorch
- pandas
- rdkit
- scipy
- seaborn
- sklearn
- tensorboard
- tqdm
- imbalanced-learn

### Quick Start
```sh
% python -m venv venv                    # create python virtual env
% source ./venv/bin/activate             # activate venv
% pip install -r requirements.txt        # install requirements (confirmed to work with CUDA 11.1 on ubuntu 18.04)

% python preprocess_atom2id.py           # calculate to adjacent matrix, Coulomb matrix, etc.
...
100%|█████████████████████████████████████| 12336/12336 [07:28<00:00, 27.49it/s]

% python train_pyrfume_lpe.py            # train and evaluate the LGNN model with pyrfume dataset.
...
min loss at 12: 0.62    max auc at 12: 0.58:   2%|▊                                    | 14/600 [21:44<14:48:30, 90.97s/it]]
...
```
### Train the pure GNN models
**Command line**
```python train_pyrfume.py --out_dir ../GNN-Coulomb --gnn_matrix coulomb```

**Parameters**:
- ```out_dir```: the output folder during training; ```${out_name}$```, please assgin the output folder in ```${out_name}$```
- ```gnn_matrix```: the matrix that we want to model; please choose from ```adjacent``` and ```coulomb```

### Train the conventional classifier with fingerprints
The scripts are located in folder ```baselines```.
**Command line**
```python run_fgs.py --fp mordreds --model knn```

**Parameters**:
- ```fp```: the fingerprints that we want to model; please choose from ```mordreds```, ```bfps```, and ```cfps```
- ```model```: the classifier that we want to train; please choose from ```knn```, ```rf```, ```gb```, ```smote-knn```, ```smote-rf```, and ```smote-gb```; ```knn``` refers to K- Nearest Neighbor, ```rf``` refers to Random Forest, ```gb``` refers to Gradient Boosting, and ```smote``` refers to Synthetic Minority Oversampling Technique.

