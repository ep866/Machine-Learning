# Machine-Learning Class


#### Assignment 3: Machine Learning Clustering

---------
* KMeans
* Dataset: Amazon reviews
* Task: Amazon recommendation system
* Report is in the Jupyter notebook
* View clusters_size.png for clusters size
* View clusters.txt for a sample of the clusters

---------


#### Assignment 2: Machine Learning Regression

---------
* Multiple regression
* Dataset: Salaries
* Task: Predict salary
* Report is in the Jupyter notebook

---------


#### Assignment 1: Machine Learning Classification

---------
* Binary Classifier
* Dataset: Amazon reviews
* Task: Predict if a review will be helpful

---------
### Model 1 / take 1 / Logistic Regression

The original Amazon dataset contains (455000, 13) samples

The features for this model are derived from a Bag of Words and additional parsing of the text of the review and dataset. The final dataframe contains (455000, 131088) samples x features as a sparse matrix.

The features I added besides the bag of words are:

|  featureName        |  feature |
|---------------------|----------|
|   nWords            | Number of words |
|   nChar             | Number of characters |
|   wordCharRatio     | Ratio between words and characters |  
|   nUpper            | Number of Uppercase letters |
|   upperRatio        | Ratio of upper to lowercase |
|   longestWord       | Longest word |
|   avgWordLen        | Average word length |
|   exclamationPoint  | Exclamation points number |  
|   punctCount        | Punctuation count |
|   punctRatio        | Ratio of punctution to characters |
|   nReviewsProducts  | Number of reviews for the product |
|   nReviewsRatio     | Ratio of number of reviews to product |
|   nRepeatUsers      | Number of users who left more than one review |
|   nUniqueUsers      | Number of unique users |
|   score				  | score |
|   time				  | time |


## The model's peformance on the test data:

##### ROC plot:
![](plots/model_1/ROC_test.png)

##### Confustion Matrix & Report:
![](plots/model_1/Matrix_test.png)

##### Measures:
![](plots/model_1/Measures_test.png)

##### Confusion Matrix Image:
![](plots/model_1/Matrix_pic_test.png)


## The model's peformance on the train data:


##### ROC plot:
![](plots/model_1/ROC_train.png)

##### Confustion Matrix & Report:
![](plots/model_1/Matrix_train.png)

##### Measures:
![](plots/model_1/Measures_train.png)

##### Confusion Matrix Image:
![](plots/model_1/Matrix_pic_train.png)


## The model's peformance on the test data without the bag of words:
I though it very curious how the model performs sans the BOW

##### ROC plot:
![](plots/model_1/model_noBOW/noBag_model.png)

##### Confustion Matrix & Report:
![](plots/model_1/model_noBOW/noBag_model3.png)

##### Confusion Matrix Image:
![](plots/model_1/model_noBOW/noBag_model2.png)



# Distribution, historgam, and descriptive stats of features

##### nWords:
![](plots/model_1/dist_nWords.png)

##### nChar:
![](plots/model_1/dist_nChar.png)

##### wordCharRatio:
![](plots/model_1/dist_wordCharRatio.png)

##### nUpper:
![](plots/model_1/dist_nUpper.png)

##### upperRatio:
![](plots/model_1/dist_nUpperRatio.png)

##### longestWord:
![](plots/model_1/dist_longestWord.png)

##### avgWordLen:
![](plots/model_1/dist_avgWordLen.png)

##### exclamationPoint:
![](plots/model_1/dist_exclamationPoint.png)

##### punctCount:
![](plots/model_1/dist_punctCount.png)

##### punctRatio:
![](plots/model_1/dist_punctRatio.png)

##### nRepeatUsers:
![](plots/model_1/dist_nRepeatUsers.png)

##### nUniqueUsers:
![](plots/model_1/dist_nUniqueUsers.png)

##### score:
![](plots/model_1/dist_Score.png)





