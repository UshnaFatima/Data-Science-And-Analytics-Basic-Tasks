# IMPORTING LIBRARIES 
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split



# IMPORTING DATASET
dataset = pd.read_csv('C:/Users/Hp/Desktop/python(datascience)/Python Project/Sentimental Analysis/Twitter analysis/Sentiment.csv')
print(dataset)

dataset = dataset[['text','sentiment']]
print(dataset)

dataset = dataset[dataset['sentiment']!= 'Neutral']
print(dataset)



# DATA PREPROCESSSING 
train,test = train_test_split(dataset,test_size=0.1)
print(train)
print(test)
print(train['text'][9720])
pattern = r"(#\w+)|(RT\s@\w+:)|(http.*)|(@\w+)"
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Cleaning 
def Clean_text(data):
    tweets = []
    sentiments = []
    for index,row in data.iterrows():
        sentence = re.sub(pattern,'',row.text)
        words = [e.lower() for e in sentence.split()]
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
        words = ' '.join(words)
        tweets.append(words)
        sentiments.append(row.sentiment)
    return tweets,sentiments

train_tweets,train_sentiments = Clean_text(train)
final_data = {'tweets':train_tweets,'sentiments':train_sentiments}
processed_data = pd.DataFrame(final_data)
print(processed_data)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
processed_data['sentiments'] = labelencoder.fit_transform(processed_data['sentiments'])
print(processed_data)

# converting words to vector 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,3))
cv.fit(processed_data['tweets'])

X_train = cv.transform(processed_data['tweets'])
print(X_train.shape)
print(X_train.toarray())

target = processed_data['sentiments'].values
print(target)




# SENTIMENTAL ANALYSISI (MODEL BUILDING)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
print(classifier.fit(X_train.toarray(),target))

test_tweets,test_sentiments = Clean_text(test)

data_test = {'tweets':test_tweets,'sentiments':test_sentiments}
final_test_data = pd.DataFrame(data_test)

X_test = cv.transform(final_test_data['tweets'])
print(X_test.shape)

y_pred = classifier.predict(X_test.toarray())
final_test_data['sentiments'] = labelencoder.fit_transform(final_test_data['sentiments'])
print(final_test_data)

actual_values = final_test_data['sentiments'].values
print(actual_values)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,actual_values))

print(final_test_data)
print(X_test.toarray()[0])
print("Predicted Sentiment:",classifier.predict(X_test.toarray()[[0]]))

