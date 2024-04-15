# blood_cell_classification


Dataset involved in this project:

> https://drive.google.com/drive/folders/1Ki4-020vszEdzjNAOPx5KHjTNjndhYER?usp=sharing


## Project Overview

Status:
1. Getting/choise the dataset
2. Structure the project
3. Exploring the data and start of the EDA
4. Pre-processing
5. Baseline Model

  > presentation of the project: 
  > https://docs.google.com/presentation/d/1CLpYXndVcSCMrxocgIsOs3Yk3uc-7qbWYqRG9Q8c9nE/edit#slide=id.g2c04e608fb2_0_2
6. Improving the model
7. Comparing with others models

_8. Improving the models to get better_

---------------

## Walkthrough Demo

Afther cloning the repository, you can install the enviroment requires to be able to run the project

> file is on the root dir under the name environment.yml

To test, you can enter in each script in the notebook's folder.

---------------
## Future ideas

~~- Sample the data to 2000 images~~
- Find a better logistic regression classification (c-value, solver and penality)

~~- Test to train/fit a NN model~~

- Test to train/fit a RCNN model

...
---------------


<!-- ### Problem Area
In the field of healthcare, the project focuses on addressing the challenges associated with identifying white blood cells from microscope images. The difficulty lies in accurately categorizing these cells, which can lead to potential misinterpretations and impact patient outcomes. -->

<!-- ### Those affected
Healthcare professionals, including Medical Laboratory Technologists, Diagnostic Cytology Technologists, Clinical Genetics Technologists, Medical Laboratory Assistants, and Oncologist Doctors, are directly affected. These individuals encounter the daily task of analyzing white blood cells and would benefit from a more efficient and accurate identification process. -->

<!-- ### Proposed Data Science solution
The proposed solution involves leveraging machine learning (ML) to develop an algorithm capable of identifying white blood cells based on visual characteristics. By training the model on a dataset of previous cell images, the system aims to provide valuable assistance to healthcare professionals in the categorization process. -->

<!-- ### Impact of your solution
The project's impact extends to both societal and business domains within the healthcare sector. By enhancing the accuracy and speed of white blood cell identification, the solution has the potential to improve healthcare diagnostics, leading to cost savings and better patient outcomes. -->

<!-- ### Description Dataset

The dataset for training the ml model comprises a collection of microscope images featuring various white blood cell phenotypes. Each image is labeled with the corresponding cell category(in this project we have a directory for each one), allowing the model to learn and generalize patterns for accurate identification. The dataset aims to represent the diversity of white blood cell characteristics encountered in real-world healthcare scenarios.

The distribution from the data clases:
![Image Alt text](/figures/screenshot_distribution.png) -->
---------------

## Project structure

```bash
── README.md
├── data
├── environment.yml
├── figures
├── models
│   ├── logistic_regression_baseline_model.pkl
│   ├── logistic_regression_model_bestC01_lbfgs_l2.pkl
│   └── vgg19-script06.h5
├── notebooks
│   ├── 0-NN-test-model.ipynb
│   ├── 01-EDA.ipynb
│   ├── 02-pre-processing.ipynb
│   ├── 03-baseline-model.ipynb
│   ├── 04-baseline-model-improved.ipynb
│   ├── 05-CNN-model-resnet50.ipynb
│   ├── 06-CNN-model-vgg19.ipynb
│   ├── 07-CNN-model-resnet50-improved.ipynb
│   ├── first_test
│   │   ├── Model.h5
│   │   └── test_script.ipynb
├── reports
└── src
    └── hello_opencv.ipynb
    └── hemato.jpeg
```

---------------

### Project Flow

Inside the notebooks dir we have our analysis process, which consist with jupyter notebooks that follow a sequence.

In their names we have the sequence and what we try/accomplish in each.

```bash
├── notebooks
│   ├── 0-NN-test-model.ipynb
│   ├── 01-EDA.ipynb
│   ├── 02-pre-processing.ipynb
│   ├── 03-baseline-model.ipynb
│   ├── 04-baseline-model-improved.ipynb
│   ├── 05-CNN-model-resnet50.ipynb
│   ├── 06-CNN-model-vgg19.ipynb
│   ├── 07-CNN-model-resnet50-improved.ipynb
```

---------------

### Credits & References

- Otto Spigariol (ottospiga@gmail.com)


