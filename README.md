# chemLGNN

This is the workspace for odor chemical recognition by laplacian graph neural network.

### Dataset
#### Data source
All comes from pyrfume-data (https://github.com/pyrfume/pyrfume-data). After data cleaning, there exists ```11863``` molecule-odor descriptors pairs, which is two fold higher than Google's (https://arxiv.org/abs/1910.10685) with ```~5000``` pairs. The folder of ```pyrfume_models3_sois``` mainly includes the preprocessed datasets with:
- Adjacent matrix
- Coulomb matrix
- Atom array
- Eigenvectors and eigenvalues of adjacent matrix
- Eigenvectors and eigenvalues of Coulomb matrix

#### Data plit
The dataset was split by second order iterative stratification, same as Google's split method (https://arxiv.org/abs/1910.10685), to handle the imbalanced distribution of odor descriptors with train/val/test of  8/1/1.

### Example
1. Please run ```python preprocess_atom2id.py``` to calculate to adjacent matrix, Coulomb matrix, etc.
2. Please run ```python train_pyrfume_lpe.py``` to train and evaluate the LGNN model with pyrfume dataset.

### To do list
Baseline models ...
