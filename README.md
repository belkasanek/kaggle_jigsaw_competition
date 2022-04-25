# [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
 The task was about to identify six different type of toxicity in comments of  Wikipedia. I trained bidirectional LSTM model with  glove embedding vectors pretrained on twitter data. To extract information from sequential output of LSTM model max and average global pooling were used. They were concatenated with last hidden state of model. The dense layer with sigmoid activation were used to make prediction. I trained model using 5 fold cross validation with early stopping when validation error stop decreasing for 3 epoch. I averaged the models prediction on test set from each fold. Because of the toxic nature of data there is a lot of misspelling. For such words I tried to find word from dictionary with the smallest levenshtein distance to the misspelled one. I also used classical approach with usage of Bag of Word weighted with TF-IDF on characters, words and bi-grams for training Logistic Regression models. I averaged prediction of those approaches and get result below.
 
|Best score|My best score|place on the leaderboard| 
|---|---|---|
|0.9885|0.9866| top 5%|
