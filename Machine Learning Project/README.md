# Machine Learning Project

In this project, I have implemented the task of **sentiment analysis on text data**. To this end following steps have been taken.

## Data Preparation

Please download ***dataset.csv*** and ***dataset2.csv*** files using the following links:
- https://drive.google.com/file/d/15JJ6ZysFM57tlUjXo2nHVhkGwePbVMVV/view?usp=sharing
- https://drive.google.com/file/d/1uykBJxWH5v5BsSuuwM0r9WLiKWQrDiDJ/view?usp=sharing

## Phase 1 (File: ML_project_phase1.ipynb)
In this phase, I preprocessed the data in file ***dataset.csv***, containing a comment on social media as text and two sentiment labels, 0 as negative and 1 as positive.
These text data are prepared in three different modes respectively:
 1. With no prepocessing
 2. With lower-case standardization and removal of numbers and additional characters
 3. With preprocessing steps of the second mode, plus stop-word removal, lemmatization, and stemming

These 3 modes are then used in classification via **SVM**, **kNN**, and **Logistic Regression** using two vecotrization approaches: **Bag of Words** (bow) and **Word2Vec**. 
In the next part, K-fold Cross Validation is applied to find the models' best hyperparameters. Best models in each mode are also stored as pickle files.
In conclusion, overall, the third preprocessing mode is a better option, as it lightens computations with only minimal negligible performance degradations in some cases. Also, Word2Vec representation yields better results as shown in the notebook. 
In the last part, a simple Multi-layer Perceptron (MLP) is also implemented and the results are obtained.
## Phase 2 (File: ML_project_phase2.ipynb)
 In this phase, the same data is analyzed for clustering using **K-means**, **Gaussian Mixture Model**, and **MiniBatch K-means**. Data representations are Word2Vec representations with full preprocessing from the previous part.
 Results are compared both visually and quantitatively. For visual evaluation, clustering results are shown using PCA in 2 dimensions on the data. Also, the following clustering metrics are reported:
 

 - Purity
 - Homogeneity
 - Completeness
 - V-Measure
 - Adjusted Rand
 - Adjusted Mutual Info 

As reported by these metrics, K-means yields the best results. So, K-means with 3 clusters is performed and sampled comments from each cluster are shown. Considering the samples, the three clusters correspond to mixed comments, negative comments, and positive comments respectively. 

In the last part, data from ***dataset2.csv***, which are comments for an online shop, are used to fine-tune the previous MLP and logistic regression models in phase 1. Specifically, using this strategy, the MLP model yields a very satisfyin performance on the new dataset.



