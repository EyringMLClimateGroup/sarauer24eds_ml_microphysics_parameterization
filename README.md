# A physics-informed machine learning parameterization for cloud microphysics in ICON
This repository contains the code for the physics-informed machine learning parameterization for cloud microphysics in ICON. The simulation data used to train and evaluate the machine learning algorithms was generated with the ICON model. 
The corresponding paper is currently under Review in Environmental Data Science
> Sarauer, Ellen, et al. "A physics-informed machine learning parameterization for cloud microphysics in ICON."

[![DOI](https://zenodo.org/badge/855790728.svg)](https://zenodo.org/badge/latestdoi/855790728)



If you want to use this repository, start by executing
```
conda env create -f environment.yml
conda activate sarauer_ml_mig
```

## Repository content
- [models](models): contains the trained ML models "Microphysics Trigger Classifier" and "Microphysics Regression".
- [notebooks](notebooks): contains notebooks for data exploration.
- [scripts](scripts): contains batch scripts on how to submit scripts in src folger to DKRZ with slurm + coarse-graining script `coarse-graining.sh`.
- [src](src): contains all important scripts for the pipeline.
    - preprocessing: `preprocess_classifier.py` and `preprocess_regression.py`
    - build models: `build_classifier_model.py` and `build_regression_model.py`
    - training: `train_classifier.py` and `train_regression.py`
    - postprocessing and explainability: `postprocess_classifier_and_explain.py` and `postprocess_regression_and_explain.py`

## Data
To fully reproduce the results it is first necessary to have access to accounts on [DKRZ/Levante](https://docs.dkrz.de/). The source code is available on the GitLab of the DKRZ (https://gitlab.dkrz.de/icon/icon-mpim) under a BSD 3-clause license (https://gitlab.dkrz.de/icon/icon-mpim/-/tree/master/LICENSES). The simulations were performed with the branch feature-nextgems-aerosol-microphysics at commit 260364f1.

## Important packages
- all models are trained with Pytorch (https://github.com/pytorch/pytorch) and using Sklearn (https://github.com/scikit-learn/scikit-learn)
- code related to the explainability of model predictions uses the Shap library (https://github.com/shap/shap)
