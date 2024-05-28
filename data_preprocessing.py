from global_imports import *

#importing the training data
imdb_data=pd.read_csv('IMDB Dataset.csv')


## Train-Test Split................
## train dataset
train_reviews=imdb_data.review[:40000]
train_sentiments=imdb_data.sentiment[:40000]
#test dataset
test_reviews=imdb_data.review[40000:]
test_sentiments=imdb_data.sentiment[40000:]
print(f"Training reviews data shape {train_reviews.shape} /t training sentiments shape {train_sentiments.shape}")
print(f"Test reviews data shape {test_reviews.shape} /t test sentiments shape {test_sentiments.shape}")


#Tokenization of text
tokenizer=ToktokTokenizer()

#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')
#set stopwords to english
stop=set(stopwords.words('english'))


#Removing the noisy text
def denoise_text(text):
    ## Removing the html strips
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    ## Removing the square brackets
    text = re.sub('\[[^]]*\]', '', text)

    ## function for removing special characters
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)

    #Stemming the text
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

imdb_data['review']=imdb_data['review'].apply(denoise_text)



### Stopwords Removal
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

imdb_data['review']=imdb_data['review'].apply(remove_stopwords)



## Data Normalization
norm_train_reviews=imdb_data.review[:40000]
norm_test_reviews=imdb_data.review[40000:]



### Bag of words
#Count vectorizer for bag of words
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
#transformed train and test reviews
cv_train_reviews=cv.fit_transform(norm_train_reviews)
cv_test_reviews=cv.transform(norm_test_reviews)

print('BOW_cv_train:',cv_train_reviews.shape)
print('BOW_cv_test:',cv_test_reviews.shape)
#vocab=cv.get_feature_names()-toget feature names

##### https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/notebook

