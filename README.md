# Oil-Spill-Detection-Model

#### -- Project Status: [Completed]

## Project Objective
The purpose of this project is to build a machine learning model to detect oil spill from satellite image using the processed dataset provided from Kaggle. In this project, i look at multiple different data augmentation techniques (SMOTE, ADASYN, etc) for imbalanced dataset and different supervised learning algorithms to improve the model performance. 

## Project Description
**About the dataset** The dataset was developed by starting with satellite images of the ocean, some of which contain an oil spill and some that do not. Images were split into sections and processed using computer vision algorithms to provide a vector of features to describe the contents of the image section or patch.
The task is, given a vector that describes the contents of a patch of a satellite image, then predicts whether the patch contains an oil spill or not, e.g. from the illegal or accidental dumping of oil in the ocean. There are two classes and the goal is to distinguish between spill and non-spill using the features for a given ocean patch. 

Non-Spill: negative case, or majority class.
Oil Spill: positive case, or minority class.
[Source](https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection)

### Project Pipeline :
1. Data processing and exploration: In this section, I performed data cleaning to remove NaN vaues and features that contains only zero values. I also checked for imblance classes and the data distribution as shown in the table below

| Category       | Total values  | Percentage(%)  |
| -------------  |:-------------:| --------------:|
| Oil spill      | 41            | 4.38           |
| No oil spill   | 896           |  95.62         |

![alt text]([https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true](https://github.com/avtnguyen/Oil-Spill-Detection-ML-Model/blob/main/Label_distribution.png))

3. 
4. Which states have the most dogs in shelters and what breeds/types those are? Does it have anything to do with the population in each states?
5. Are there any trends in the types/breeds of dogs being brought into shelters?

### Methods Used
* Data Cleaning and Wrangling
* Data Analysis
* Data Visualization

### Technologies
* Pandas
* Numpy
* Seaborn and Pyplot
* Colab

## Needs of this project
- Data exploration/descriptive statistics
- Data processing/cleaning
- Writeup/reporting

## Getting Started
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is can be dowloaded from [Kaggle](https://www.kaggle.com/datasets/whenamancodes/dog-adoption), [US population census](https://www.census.gov/newsroom/press-kits/2019/national-state-estimates.html)
3. Data processing and visualization scripts are being kept [here](https://github.com/avtnguyen/Dog_Adoption_DataAnalysis/blob/main/Dog_Adoption_Data_Analysis_Project.ipynb)

## References:
* Raw data is obtained from [Kaggle](https://www.kaggle.com/datasets/whenamancodes/dog-adoption), [US population census](https://www.census.gov/newsroom/press-kits/2019/national-state-estimates.html)
* For further reading about dog shelters: [Challenges in animal shelters](https://globalnews.ca/news/8997583/canadian-animal-shelters-challenges/),
[Popular breeds in dog shelters](https://rescuedoghome.com/why-are-there-so-many-pit-bulls-and-chihuahuas-in-shelters/#Chihuahuas_in_Shelters)

## Contributing Members

**Team Leads (Contacts) : [Anh Nguyen ](https://github.com/avtnguyen)**

## Contact
* Feel free to contact team leads with any questions or if you are interested in contributing!
