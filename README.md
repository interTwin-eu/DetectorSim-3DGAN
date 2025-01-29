# 3DGAN model for fast particle detector simulation

### 3D Generative Adversarial Network for generation of images of calorimeter depositions
This project is based on the prototype 3DGAN model developed at CERN and is developed on PyTorch Lightning framework. Can also be found in the following repositories in different ML framework versions: 
- Keras on TF1: [https://github.com/svalleco/3Dgan/tree/Anglegan/keras](https://github.com/svalleco/3Dgan/tree/Anglegan/keras)
- Keras on TF2 (Accelerated3DGAN): [https://github.com/CERN-IT-INNOVATION/3DGAN](https://github.com/CERN-IT-INNOVATION/3DGAN)

## Dataset
The data used for the training and validation processes of the model are 3D images representing calorimeter energy depositions and are publicly available in different formats:
### Restricted Energy Data 100-200 GeV
- compressed format: [https://zenodo.org/records/3603086#.ZEJOTs5Byqi](https://zenodo.org/records/3603086#.ZEJOTs5Byqi)
- HDF5: [https://cernbox.cern.ch/s/3rK5UeRzRR3Kbnu](https://cernbox.cern.ch/s/3rK5UeRzRR3Kbnu)
- [tfrecords](https://cernbox.cern.ch/files/link/public/DEUSrqXGVLUwpK2?tiles-size=1&items-per-page=100&view-mode=resource-table&sort-by=name&sort-dir=asc) (used for the Accelerated3DGAN version)


## Related work
1. Khattak, G.R., Vallecorsa, S., Carminati, F. et al. Fast simulation of a high granularity calorimeter by generative adversarial networks. Eur. Phys. J. C 82, 386 (2022). DOI:  [https://doi.org/10.1140/epjc/s10052-022-10258-4](https://doi.org/10.1140/epjc/s10052-022-10258-4)
2. Physics Validation of Novel Convolutional 2D Architectures for Speeding Up High Energy Physics Simulations, Florian Rehm, Sofia Vallecorsa, Kerstin Borras, Dirk Krücker. Paper published at vCHEP2021 conference. DOI: [https://doi.org/10.48550/arXiv.2105.08960](https://doi.org/10.48550/arXiv.2105.08960)
