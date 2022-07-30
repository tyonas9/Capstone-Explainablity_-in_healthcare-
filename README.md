# Capstone-Explainablity_in_healthcare-

Even though this project don't seem to involve EDA, in the following below we noted what we've done to setup/recreate the environment based on the instructions in the github [https://github.com/DeepPathology/MITOS_WSI_CMC] repositories and go through the notebooks to get the results [https://github.com/DeepPathology/MITOS_WSI_CMC/tree/master/results]

## Overview

The dataset contains two main parts:

### Data set variant evaluation

The fully annotated dataset of breast cancer whole slide images is available as open-source [https://www.nature.com/articles/s41597-020-00756-z]. The original whole slide images have been cut into small tiles due to the high-resolution. Each tile has been expertly analyzed and annotated. A computer vision model that provides the correct output with high-probability is described in the documentation.

This folder contains the evaluation for all variants, i.e. the manually labelled (MEL), the the object-detection augmented manually expert labelled (ODAEL), and the clustering- and object detection augmented manually expert labelled (CODAEL)[] variant.

Went through the evaluation process based on the "Evaluation.ipynb"[https://github.com/DeepPathology/MITOS_WSI_CMC/blob/master/Evaluation.ipynb] to get results of the data set variants based on a one-and two-stage-detecor (i.e. mitosis detector)

Main results of the data set variants based on a one- and two-stage-detector can be found in Evaluation.ipynb.

NOTE: Alternative training set for the TUPAC16 auxilary mitotic figure data set, instructions on how to stitch images to facilitate labeing and later training, sqlite databases to store annotations, baseline training of RetinaNet, and Crossvalidation resulst are also found here[https://github.com/DeepPathology/TUPAC16_AlternativeLabels]

## Setting up the environment

Followed the Setup.ipynb [https://github.com/DeepPathology/MITOS_WSI_CMC/blob/master/Setup.ipynb] to setup the dataset of 37GB size WSI ( whole Slide Images) from figshare. We then used the data loaders provided in this repository to get a visual impression of the dataset. We also used annotation tool our annotation tool SlideRunner(https://github.com/maubreville/SlideRunner) to look at the WSI images and also the annotated training and testing datasets. 


## Training notebooks

followed and run the following notebooks below to train a RetinaNet model on the respective dataset variants . The training process can be seen in the notebooks for the respective dataset variants:

 RetinaNet-CMC-MEL.ipynb (RetinaNet-CMC-MEL.ipynb) [https://github.com/DeepPathology/MITOS_WSI_CMC/blob/master/RetinaNet-CMC-MEL.ipynb]

RetinaNet-CMC-ODAEL.ipynb (RetinaNet-CMC-ODAEL.ipynb)[https://github.com/DeepPathology/MITOS_WSI_CMC/blob/master/RetinaNet-CMC-ODAEL.ipynb]

RetinaNet-CMC-CODAEL.ipynb (RetinaNet-CMC-CODAEL.ipynb) [https://github.com/DeepPathology/MITOS_WSI_CMC/blob/master/RetinaNet-CMC-CODAEL.ipynb]