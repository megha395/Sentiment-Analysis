from feature_extraction import *

### Building logistic regression model for both bag of words and tfidf features
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
## Fitting the model for Bag of words
lr_bow=lr.fit(cv_train_reviews,train_sentiments)
print(f"Bag of words model \n {lr_bow}")

#Fitting the model for tfidf features
lr_tfidf=lr.fit(tv_train_reviews,train_sentiments)
print(f"TF-IDF model \n {lr_tfidf}")


