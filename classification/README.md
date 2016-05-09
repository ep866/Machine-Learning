#### Assignment 1: Machine Learning Classification

---------
#####Final Model: Classification-updated-v3-FINAL
* Results, Report & Visualizations of final model in Jupyter Notebook
* Model: Stochastic Gradient Descent / Logistic Regression
* Binary Classifier
* Dataset: Amazon reviews
* Task: Predict if a review will be helpful

#### Classification results from final model

## The model:

##### Hashing Vectorizer - Bag of Words:
![](plots/model_final/vectorizer.png)

##### Stochastic Gradient Descent / Log Loss Function:
![](plots/model_final/model.png)


## The model's performance on the test data:

##### ROC plot & Confustion Matrix:
![](plots/model_final/test_roc.png)

##### Evaluation measures:
![](plots/model_final/test_eval.png)

##### Confusion Matrix:
![](plots/model_final/test_matrix.png)



## The model's performance on the train data:


##### ROC plot & Confustion Matrix:
![](plots/model_final/train_roc.png)

##### Evaluation measures:
![](plots/model_final/train_eval.png)

##### Confusion Matrix:
![](plots/model_final/train_matrix.png)


---------

#### Initial classification results from model_1.py

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





