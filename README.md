# brain-scan-segmentation
Northeastern University | MATH4570 | Spring 2024
![image](https://github.com/maxseidel/brain-scan-segmentation/assets/22263705/8d1b909a-c32b-49bd-b323-35c5c107f782)

# Brain CT Image Hemorrhage Segmentation by Artificial Neural Network

This repository houses the project "Brain CT Image Hemorrhage Segmentation by Artificial Neural Network," a collaboration with Zeta Surgical, aimed at advancing the field of medical imaging through machine learning and artificial neural networks. Zeta Surgical, founded by Harvard graduates, is committed to democratizing access to accurate, safe, and fast image-guidance, enhancing point-of-care treatments, and enabling new treatments for emergencies and bedside procedures.

## Project Overview

The focus of this project is on the segmentation of hemorrhages in brain CT images using machine learning and artificial neural network techniques. The dataset provided by Zeta Surgical includes various CT scan slices, each depicting hemorrhages classified into different types: intraparenchymal, intraventricular, subarachnoid, subdural, epidural, and a category for images with multiple sources of bleeding. 

### Objectives

- To develop mathematical models for the classification, regression, and segmentation of brain hemorrhages.
- To apply machine learning and artificial neural network techniques using Python and TensorFlow.
- To investigate the provided labeled dataset for better accuracy in hemorrhage detection and segmentation.

## Dataset

The dataset consists of brain CT scan slices, divided into several types of hemorrhages. Each bottom-level directory contains 4 subdirectories (`brain_bone_window`, `bone_window`, `max_contrast_window`, `subdural_window`) for different CT rendering types, alongside raw dicom files in the `dcms/` subfolder. The dataset is organized as follows:

- `normal/`: Images without any hemorrhaging.
- `multi/`: Images with multiple hemorrhages.
- `Results_{Type} Hemorrhage Detection_{Datetime}.csv`: Six CSV files containing segmentation results, separated by hemorrhage subtype.
- `Segmentation Glossary.rtf`: Documentation on the format of the CSV files.
- `flagged-cases.txt`: A list of cases flagged for incorrect labeling, including a note on corrupt files.


## Installation and Usage

Before running the project, ensure you have Python and TensorFlow installed. The code is designed to be executed in a Python environment. Follow these steps to set up the project:

1. Clone this repository to your local machine.
2. Install the required dependencies.
   1. ```python3 -m venv .venv```
   2. ```. .venv/bin/activate```
   3. ```pip3 install -r requirements.txt  ```
3. Download the dataset from the provided links in canvas and place it in the appropriate directory structure as mentioned above.
4. Run the main script to start the segmentation process.

## Contact

For further information, please contact:

- Raahil Sha, CTO at Zeta Surgical ([Contact Information](https://www.zetasurgical.com/))

## Acknowledgments

This project is made possible through the collaboration with Zeta Surgical and the efforts of numerous students and contributors focused on improving medical imaging technologies.

---
