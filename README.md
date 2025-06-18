## Enhancement of Logistic Regression Algorithm for Email Spam Detection

   ```bash
https://doi.org/10.36948/ijfmr.2024.v06i06.32374
   ```

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Technologies and Tools](#technologies-and-tools)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [References](#references)

## Introduction
This project focuses on enhancing the logistic regression algorithm applied in email spam detection. By utilizing techniques like TF-IDF (Term Frequency-Inverse Document Frequency), Recursive Feature Elimination (RFE), and Principal Component Analysis (PCA), we aim to improve the model's accuracy, efficiency, and handling of class imbalance.

## Problem Statement
1. **Class Imbalance:**
   Logistic regression tends to be biased towards the larger group, leading to less accurate predictions for the minority class, which can result in financial losses and reputational damage (Zhang L & Geisler T et al., 2021). Class imbalance complicates the model's specification and accuracy (Suaad & Intesar, 2023).

2. **Large Datasets:**
   Logistic regression models can struggle to accurately classify spam in larger datasets (Bilge & Bariye, 2019). Traditional regression techniques are generally effective only for smaller datasets, highlighting the need for enhancement.

3. **Overfitting:**
   Logistic regression is prone to overfitting, especially if the model contains too many irrelevant features or is too complex relative to the data (Awan, 2023; Olaoye et al., 2024). This results in poor generalization to unseen data.

## Objectives
1. **Improve Feature Representation with TF-IDF:** Enhance logistic regression's performance on imbalanced datasets by refining textual feature representation, emphasizing terms significant to the minority class.

2. **Enhance Efficiency with Recursive Feature Elimination (RFE):** Utilize RFE to reduce data complexity, focus on the most critical features, and lessen the computational load, especially for large datasets.

3. **Reduce Overfitting with Principal Component Analysis (PCA):** Streamline logistic regression by reducing feature space dimensions, focusing on relevant data aspects, and eliminating noisy features.

## Technologies and Tools
- **Programming Language:** Python
- **Libraries:** scikit-learn, pandas, numpy, matplotlib
- **Algorithm Enhancements:** TF-IDF, RFE, PCA

## Data
The dataset used for this project includes real-world email data for spam classification. It has been cleaned and preprocessed to improve training efficiency and accuracy.

## Installation
To set up this project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-detection-logistic-regression.git

## Flow of the System

- Create/Choose Designated Gmail Account
- Create Gmail API
- Set up Gmail API
- Get the credentials and set up the python file
- Connect Gmail API to the front-end (fetch button)
- Create test function for the algorithm to make sure its working before connecting it to the front-end
- Connect the algorithm to the front-end

## Research Team
# This thesis was prepared and researched by:
Carlos, Vince Anthony S.
Pancho, John Cedric C.

"# LogisticRegression" 
