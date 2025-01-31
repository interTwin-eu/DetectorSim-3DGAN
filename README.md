# This branch contains 3DGAN developed in Tensorflow/Keras
### Credit: https://github.com/svalleco/3Dgan/tree/Anglegan/keras 

# Training with variable angle data 
AngleArch3dGAN.py is the architecture and AngleTrain3dGAN.py is the training script.

The weights dir is used to store weights from training. If weights for different trainings are to \
be saved then --name can be used at command line to identify the training.


## Dataset
The data used for the training and validation processes of the model are 3D images representing calorimeter energy depositions and are publicly available in different formats:
- compressed format: [https://zenodo.org/records/3603086#.ZEJOTs5Byqi](https://zenodo.org/records/3603086#.ZEJOTs5Byqi)
### Restricted Energy Data 100-200 GeV
- HDF5: [https://cernbox.cern.ch/s/3rK5UeRzRR3Kbnu](https://cernbox.cern.ch/s/3rK5UeRzRR3Kbnu)
- [tfrecords](https://cernbox.cern.ch/files/link/public/DEUSrqXGVLUwpK2?tiles-size=1&items-per-page=100&view-mode=resource-table&sort-by=name&sort-dir=asc) (used for the Accelerated3DGAN version)
### Full Energy Range Data 2-500 GeV
- HDF5: [https://cernbox.cern.ch/s/i7A0kfv6oqpLIZq](https://cernbox.cern.ch/s/i7A0kfv6oqpLIZq)

## Analysis
The analysis compares the GAN generated images to G4 data events. All of the scripts in this section, take a sample of G4 events and then generate GAN events with similar input conditions (primary particle energy / primary particleincident  angle). Where events are selected in bins: the primary energy bins have a +/- 5 GeV tolerance and the incident angle bins have a tolerance of +/- 0.1 rad (5.73 degree). The [utils](https://github.com/interTwin-eu/DetectorSim-3DGAN/tree/main/Lightning3DGAN/analysis/utils) directory contains a set of files with frequently used utility functions. Most of the scripts except the LossPlotsPython.py require [ROOT software](https://root.cern.ch/) to be installed. Following is a brief description and a set of instructions for all scripts in this folder:
A common feature for all the scripts is the ang (1: variable angle version 0: fixed angle version). The default is variable angle. The instructions will include only most useful parameters. Other options can be explored from parser help. 

## 2Dprojections.py

This scripts compares 2D projections for all the three planes for events from the G4 data with corressponding GAN generated events with same input values. The script can be submitted as:

python3 2Dprojections.py --gweight *generator_weights* --outdir *results/your_result_dir*

### SimpleAnalysis.py

This scripts compares the two most crucial features of the generated events: the sampling fraction and the shower shapes. The script can be submitted as:

python3 SimpleAnalysis.py --gweights *weight1 weight2* --labels label1 label2 --outdir *results/your_result_dir*

### RootAnalysisAngle.py

The scripts compares in detail different features of G4 and GAN events. The script can be submitted as:

python3 RootAnalysisAngle.py --gweights *generator_weight1 generator_weight2* --dweights *discriminator_weight1  discriminator_weight2* --labels label1 label2 --outdir *results/your_result_dir*

### LossPlotsPython.py

This script takes the loss history generated from each training, saved as a pickle file. The plots are generated using matplotlib. The script can be submitted as:

python3 LossPlotsPython.py --historyfile *path_to_loss_history* --outdir *path_to_save_results*

### SelectEpoch.py

This script select the best epoch among a predefined number of epochs. The plots also provides the epoch to epoch progress based on the sampling fraction. The script can be submitted as:

python SelectEpoch.py --gweightsdir *path_to_weights_directory* --outdir *path_to_save_results* 


## Related work
1. Khattak, G.R., Vallecorsa, S., Carminati, F. et al. Fast simulation of a high granularity calorimeter by generative adversarial networks. Eur. Phys. J. C 82, 386 (2022). DOI:  [https://doi.org/10.1140/epjc/s10052-022-10258-4](https://doi.org/10.1140/epjc/s10052-022-10258-4)
2. Physics Validation of Novel Convolutional 2D Architectures for Speeding Up High Energy Physics Simulations, Florian Rehm, Sofia Vallecorsa, Kerstin Borras, Dirk Krücker. Paper published at vCHEP2021 conference. DOI: [https://doi.org/10.48550/arXiv.2105.08960](https://doi.org/10.48550/arXiv.2105.08960)
