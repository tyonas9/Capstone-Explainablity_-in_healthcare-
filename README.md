## Explainability in Healthcare

### Capstone Project: Automating Explainability in Healthcare

Machine learning models are black boxes, and attempts to explain them aren't suitable for non-technical stakeholders. By leveraging Explainable AI (XAI) techniques one can alleviate the scrutiny and alter the assumptions behind models blackbox nature. 
In this project, we incorporate Explainable AI (XAI) techniques in medical diagnosis.

### Team Memebers

 - Rudra Bandhu
 - Yonas Tesh

### Data Lineage

Paper: https://www.nature.com/articles/s41597-020-00756-z  
Dataset:  https://github.com/DeepPathology/MITOS_WSI_CMC

### Dataset

The dataset, consisting of 21 anonymized WSIs in Aperio SVS file format, is publicly available on figshare20. Alongside, we provide cell annotations according to both classes in a SQLite3 database. For each annotation, this database provides:

- The WSI of the annotation

- The absolute center coordinates (x,y) on the WSI

- The class labels, assigned by all experts and the final agreed class. Each annotation label is included in this, resulting in at least two labels (in the case of initial agreement and no further modifications), one by each expert. The unique numeric identifier of each label furthermore represents the order in which the labels were added to the database.

### Dataset variant evaluation

The fully annotated dataset of breast cancer whole slide images is available as open-source [https://www.nature.com/articles/s41597-020-00756-z]. The original whole slide images have been cut into small tiles due to the high-resolution. Each tile has been expertly analyzed and annotated. A computer vision model that provides the correct output with high-probability is described in the documentation.

This folder contains the evaluation for all variants, i.e. the manually labelled (MEL), the the object-detection augmented manually expert labelled (ODAEL), and the clustering- and object detection augmented manually expert labelled (CODAEL)[] variant.

Went through the evaluation process based on the "Evaluation.ipynb"[https://github.com/DeepPathology/MITOS_WSI_CMC/blob/master/Evaluation.ipynb] to get results of the data set variants based on a one-and two-stage-detecor (i.e. mitosis detector)

NOTE: Alternative training set for the TUPAC16 auxilary mitotic figure data set, instructions on how to stitch images to facilitate labeing and later training, sqlite databases to store annotations, baseline training of RetinaNet, and Crossvalidation resulst are also found here[https://github.com/DeepPathology/TUPAC16_AlternativeLabels]

### Setting up the environment

Followed the Setup.ipynb [https://github.com/DeepPathology/MITOS_WSI_CMC/blob/master/Setup.ipynb] to setup the dataset of 37GB size WSI ( whole Slide Images) from figshare. We then used the data loaders provided in this repository to get a visual impression of the dataset. We also used annotation tool our annotation tool SlideRunner(https://github.com/maubreville/SlideRunner) to look at the WSI images and also the annotated training and testing datasets.

### Database Folder

- [x] sqlite database files that contain annotations

### Training notebooks

Followed and run the following notebooks below to train a RetinaNet model on the respective dataset variants . The training process can be seen in the notebooks for the respective dataset variants:

RetinaNet-CMC-MEL.ipynb (RetinaNet-CMC-MEL.ipynb) [https://github.com/DeepPathology/MITOS_WSI_CMC/blob/master/RetinaNet-CMC-MEL.ipynb]

RetinaNet-CMC-ODAEL.ipynb (RetinaNet-CMC-ODAEL.ipynb)[https://github.com/DeepPathology/MITOS_WSI_CMC/blob/master/RetinaNet-CMC-ODAEL.ipynb]

RetinaNet-CMC-CODAEL.ipynb (RetinaNet-CMC-CODAEL.ipynb) [https://github.com/DeepPathology/MITOS_WSI_CMC/blob/master/RetinaNet-CMC-CODAEL.ipynb]


### References:
- Aubreville, M., Bertram, C.A., Donovan, T.A. et al. A completely annotated whole slide image dataset of canine breast cancer to aid human breast cancer research. Sci Data 7, 417 (2020). https://doi.org/10.1038/s41597-020-00756-z
- M. Aubreville, C. Bertram, R. Klopfleisch and A. Maier (2018) SlideRunner - A Tool for Massive Cell Annotations in Whole Slide Images. In: Bildverarbeitung für die Medizin 2018. Springer Vieweg, Berlin, Heidelberg, 2018. pp. 309-314. link arXiv:1802.02347


