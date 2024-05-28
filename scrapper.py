import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
config_proto = tf.compat.v1.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
session = tf.compat.v1.Session(config=config_proto)
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
import cufflinks
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

url = "https://www.livemint.com/news/world"

browser = webdriver.Firefox()

browser.get(url)
time.sleep(0.5)

elem = browser.find_element_by_tag_name("body")

no_of_pagedowns = 5
links = []
while no_of_pagedowns:
    time.sleep(1)
    elem.send_keys(Keys.PAGE_DOWN)
    no_of_pagedowns-=1

post_elems = browser.find_elements_by_class_name("headlineSec")
newvar = []
linkselements = browser.find_elements_by_css_selector(".headlineSec > a")
for post in post_elems:
    newvar.append(post.text.encode("utf-8"))
    links.append(post.find_element_by_css_selector('a').get_attribute('href'))




artndate = []
for x in newvar:
    artndate.append(x.splitlines())

dateofart = []
for i in range(len(artndate)):
    dateofart.append(artndate[i][1][13:])

print(dateofart)


links = []
for post in post_elems:
    newvar.append(post.text.encode("utf-8"))
    links.append(post.find_element_by_css_selector('a').get_attribute('href').encode("utf-8"))







news_text = []

for news_l in links:
    browser.get(news_l)
    time.sleep(0.5)
    try:
        browser.find_element_by_link_text("Skip").click()
    except:
        print()
    time.sleep(2)
    post_link = browser.find_elements_by_class_name("mainArea")
    news_text.append(post_link[0].text.encode("utf-8"))
    print(news_text)




len(links)

 
import pandas as pd
dict = {'date':dateofart,'headline':newvar[0:40],'text': news_text, 'url': links}  
    
df = pd.DataFrame(dict) 


df.to_csv('scrapped_data.csv')

df = pd.read_csv('scrapped_data.csv')

news_text=df["text"]        
dateofart=df["date"]
newvar=df["headline"]
links=df["url"]   


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

from tensorflow.keras.models import load_model
# 
## load model
model = load_model('model2.h5')

import nltk.data
abcd = nltk.data.load('tokenizers/punkt/english.pickle')


sentiment=[]
sen_sentiment=[]
for i in news_text:
    
    seq = tokenizer.texts_to_sequences([i])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    labels = [ 'negative','positive']
    sentiment.append(labels[np.argmax(pred)])
#    print(pred, labels[np.argmax(pred)])
    sentences=abcd.tokenize(i)
    text=''
    for j in sentences:
        seq = tokenizer.texts_to_sequences([j])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        pred = model.predict(padded)
        labels = [ 'negative','positive']
        
        text=text+j+' -> '+labels[np.argmax(pred)]+'  '
        
    sen_sentiment.append(text)




# dictionary of lists  
dict = {'date':dateofart,'headline':newvar,'text': news_text, 'url': links, 'label': sentiment,'sentence_sentiment':sen_sentiment}  
    
df = pd.DataFrame(dict) 


df.to_csv('result.csv')     


