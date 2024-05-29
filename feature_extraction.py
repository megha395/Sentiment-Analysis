from data_preprocessing import *

## Data Normalization
norm_train_reviews=imdb_data.review[:40000]
norm_test_reviews=imdb_data.review[40000:]


### Bag of words:  Convert text documents to numerical vectors or bag of words.
## Count vectorizer for bag of words
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
## Transformed train and test reviews
cv_train_reviews=cv.fit_transform(norm_train_reviews)
cv_test_reviews=cv.transform(norm_test_reviews)

print('BOW_cv_train:',cv_train_reviews.shape)
print('BOW_cv_test:',cv_test_reviews.shape)

## Get feature names
# vocab=cv.get_feature_names()

## Tfidf vectorizer: Convert text documents to matrix of tfidf features.
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
tv_train_reviews=tv.fit_transform(norm_train_reviews)
tv_test_reviews=tv.transform(norm_test_reviews)
print(f"Tfidf train reviews shape {tv_train_reviews.shape}")
print(f"Tfidf test reviews shape {tv_test_reviews.shape}")


## labeling the sentient data
lb=LabelBinarizer()
sentiment_data=lb.fit_transform(imdb_data['sentiment'])
print(f"sentiment data shape is {sentiment_data.shape}")
 
## Spliting the sentiment data
train_sentiments=sentiment_data[:40000]
test_sentiments=sentiment_data[40000:]
print(f"Train Sentiment data is: {train_sentiments}")
print(f"Test Sentiment data is: {test_sentiments}")

