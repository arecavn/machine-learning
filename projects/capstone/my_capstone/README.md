# Machine Learning Engineer Nanodegree
## Project: Capstone Proposal and Capstone Project


### Problem
Job Salary Prediction.
A regression supervised learning problem.
### Dataset
I use Train_rev1.gzip from:
https://www.kaggle.com/c/job-salary-prediction/data

244,768 jobs, 12 columns with 412MB compressed size.
The dataset is highly skewed to the left, 75% of jobs have the lower salary than 25% of max salary. Have a rather big text field.

### Metric
- MAE: 5556
- Good rate: 84
### Algorithm
- Use LightGBM on H2O AI platform via XGBoost machine.
- Use Word2Vec to preprocess text fields.
### Software
- H2O cluster version:     3.14.0.3
- Python version:	3.6.2 final
- Jupyter version:            4.3.0
### Hardware
- 8 Cores
- 12G RAM
- Macbook pro
### Runtime
Around 5.2 hours.
This is not included the vectorizing 2 text fields (around 0.5 hours)
### Parameters
Learning rate:         0.03
Max leaves:            800      (max_depth: 0)
Max bins:              63
Col sample rate:       0.6
Nfolds:                3
Stopping round:        3
Stopping metrics:      MAE
