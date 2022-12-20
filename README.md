# Oil Spill Detection Model
<img src="https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/3837.jpg" width="500" align = "center">
#### -- Project Status: [Completed]

## Project Motivations:

Oil spills can have devastating environmental and economic impacts, including damage to ecosystems, wildlife, and human health. They can also disrupt economic activities, such as fishing and tourism, and harm local communities. As such, there is a need for effective tools to prevent, detect, and respond to oil spills.

One potential solution is the use of satellite imagery to monitor and classify oil spills. Satellite imagery provides a comprehensive, real-time view of large areas, and can be used to detect and map oil spills with high spatial and temporal resolution.

However, manually analyzing large amounts of satellite imagery to identify and classify oil spills is time-consuming and resource-intensive. Machine learning techniques can be used to automate this process, allowing for faster and more efficient analysis of satellite imagery.

A machine learning model for oil spill classification could be trained on a large dataset of labeled satellite images, and then used to classify new images as either containing an oil spill or not. This could greatly improve the efficiency and accuracy of oil spill detection and classification, and allow for more timely and effective response to spills.

Thus, the objective of this project is to develop an oil spill classification model from processed satellite images. By automating the analysis of satellite imagery with machine learning, it is possible to more effectively monitor and classify oil spills, enabling timely and appropriate response.

## Project Description

The Oil Spill Prediction project involves the development of a machine learning model for the classification of oil spills. The model was built using various machine learning techniques, including random forest, XGBoost, data augmentation, and hyperparameter tuning.

One of the main goals of the project was to improve the performance of the model, and this was achieved by conducting a comparative analysis and optimizing key classification metrics. As a result of these efforts, the performance of the model was improved by 25%.

The model will be used to predict the likelihood of an oil spill occurring in a given location based on the processed  satellite images. This could be used for a variety of purposes, such as environmental protection, risk assessment, and emergency response planning.

**About the dataset** The dataset was developed by starting with satellite images of the ocean, some of which contain an oil spill and some that do not. Images were split into sections and processed using computer vision algorithms to provide a vector of features to describe the contents of the image section or patch.
The task is, given a vector that describes the contents of a patch of a satellite image, then predicts whether the patch contains an oil spill or not, e.g. from the illegal or accidental dumping of oil in the ocean. There are two classes and the goal is to distinguish between spill and non-spill using the features for a given ocean patch. 

Non-Spill: negative case, or majority class.
Oil Spill: positive case, or minority class.
[Source](https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection)

### Project Pipeline :
**1. Data processing and exploration:** In this section, I performed data cleaning to remove NaN vaues and features that contains only zero values. I also checked for imblance classes and the data distribution as shown in the table below

| Category       | Total values  | Percentage(%)  |
| -------------  |:-------------:| --------------:|
| Oil spill      | 41            | 4.38           |
| No oil spill   | 896           |  95.62         |

<img src="https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/Label_distribution.png">

**2. Feature selection:** I performed feature selection based on the univariate statistical tests by computing the ANOVA F-value betwen the numerical features (e.g., f_1, f_2...) and the label target. The new dataset includes the most 25 features and f_46 because it is a categorical feature. 

**3. Splitting the dataset** to train test sets based on the following specifications: Train size: 75%, test size: 25%, stratifying based on the y label  to ensure that both the train and test sets have the same class proportion similar to the original dataset. After that, I normalized both train and test datasets using the StandardScaller() to remove the mean and scaling to unit variance. 

**4. Data augmentation**: Since the dataset is highly imbalanced, i implemented multiple data augmentation techniques to improve the quality of the dataset based on the following algorithms:

* Synthetic Minority Oversampling Technique(SMOTE): The sample in minority class is first selected randomly and its k nearest minority class neighbors are found based on the K-nearest neighbors algorithm. The synthetic data is generated between two instances in feature space. 
* Adaptive Synthetic Sampling (ADASYN): The synthetic data for minority class is generated based on the desnity distribution of the minority class. Specifically, more data is created in area with low density of minority class and less data is generated in area with high density of minority example
* SMOTE-TOMEK: Combine SMOTE and TOMEK techniqes where the oversampling technique for minority class and the cleaning using Tomek links.  
* SMOTE- ENN: Combine SMOTE and Edited Nearest Neighbours (ENN) techniques where the oversampling technique for minority class and the cleaning using ENN

Source: [Imbalanced learn](https://imbalanced-learn.org/stable/references/over_sampling.html)
 
**5. Build a simple deep learning network** and combine with multiple data augmentation techniques [See code here](https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/oil_spill_detection_deepLearningModel.ipynb)
<img src="https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/DNN_summary.png">

**6. Implement ensemble learning algorithms**, which include Random Forest, and XGBoost, and compare the model performance given the unbalanced dataset for oil spill detection [See code here](https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/oil_spill_detection_model.ipynb)

### Evaluation metrics
For imbalance dataset and classification model, the following metrics are used to evaluate the model performance:
* Precision
* Recall
* f1 score

### Results:
- Both the precision, recall and f1 scores are low in all models. This could be due to the small imbalance dataset that we have.
- Given the dataset without any resampling technique, XGBoost outperformed other algorithms.
- When data augmentation technique is implemented, the performance of Random Forrest model is improved significantly using SMOTE+TOMEK technique as shown in table below.
- More data is needed to improve the model accuracy for oil spill detection

<img src="https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/f1_scores.png" width="700" align = "center">

| model       | resample                     | precision  | recall | f1   |
| ------------|:----------------------------:| ----------:|-------:|-----:|
| DNN         | SMOTE                        |   0.267    |0.8     |0.4   |
| DNN         | SMOTE+TOMEK                  |   0.385    |0.5     |0.435 |
| RF          | SMOTE+TOMEK                  |  0.625     |0.5     |0.555 |
| RF          | SMOTE+ENN                    |   0.461    |0.6     |0.522 |
| XGBoost     | ADASYN                       |   0.5      |0.5     |0.5   |
| XGBoost     | SMOTE                        |   0.454    |0.5     |0.476 |
| DNN         | No resample                  |   0.114    |0.9     |0.2   |
| RF          | No resample                  |   0.2      |0.1     |0.133 |
| XGBoost     | No resample                  |   0.357    |0.5     |0.417 |



### Methods Used
* Data Cleaning and Wrangling
* Data Analysis
* Data Visualization
* Data Augmentation
* Machine Learning Model: Deep learning network, Random Forest, XGBoost

### Technologies
* Pandas
* Numpy
* Seaborn and Pyplot
* sklearn
* TensorFlow
* imbalanced-learn
* Colab

## Needs of this project
- Data exploration/descriptive statistics
- Data processing/cleaning
- Data modeling
- Writeup/reporting

## Getting Started
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is can be dowloaded from [this repository](https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/oil_spill.csv) or from [Kaggle](https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection)
3. Data processing and modeling scripts are being kept [here](https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/)

## References:
* https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection
* https://imbalanced-learn.org/stable/references
* https://machinelearningmastery.com/framework-for-imbalanced-classification-projects

## Contributing Members

**Team Leads (Contacts) : [Anh Nguyen ](https://github.com/avtnguyen)**

## Contact
* Feel free to contact team leads with any questions or if you are interested in contributing!
