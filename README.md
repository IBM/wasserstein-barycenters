# Wasserstein Barycenters

Wasserstein Barycenters were presented for models ensembling in the ICLR'19 paper: 

["Wasserstein Barycenter Model Ensembling"](https://arxiv.org/abs/1902.04999), Pierre Dognin, Igor Melnyk, Youssef Mroueh, Jarret Ross, Cicero Dos Santos, and Tom Sercu available on [arxiv](https://arxiv.org/abs/1902.04999) and [openreview](https://openreview.net/forum?id=H1g4k309F7)

```
@inproceedings{
dognin2018wasserstein,
title={Wasserstein Barycenter Model Ensembling},
author={Pierre Dognin and Igor Melnyk and Youssef Mroueh and Jarret Ross and Cicero Dos Santos and Tom Sercu},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=H1g4k309F7},
}
```

This github repository provides the [PyTorch](http://pytorch.org) code for computing Wassersteing Barycenters.
The code is ready for use with PyTorch 1.0.0.

# Code

To run a simple wasserstein barycenter test code, use pytorch version >=1.0.0:

```bash
python test.py
```

# Notebooks

We provide a notebook for synthetic experiments:

```bash
 jupyter ./notebooks/synthetic_experiments.ipynb
 ```
