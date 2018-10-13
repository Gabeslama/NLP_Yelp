#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline



yelp = pd.read_csv('yelp.csv')

#looking at data 
yelp.head()
yelp.info()
yelp.describe()


####Data analysis######

#new columns for number of words in in text column 
yelp['text length'] = yelp['text'].apply(len)

#text length based off number of stars (1-5)
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')
plt.show()

#occurrences for each star rating
sns.countplot(x='stars',data=yelp,palette='rainbow')
plt.show()
#dataframe to get mean of numerical columns
stars = yelp.groupby('stars').mean()

#heatmap based off correlation between cool useful funny and text length
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)
plt.show()
###NLP Classification####

yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]

X = yelp_class['text']
y = yelp_class['stars']


cv = CountVectorizer()
#fitting for countvectorization
X = cv.fit_transform(X)

####Train Test Split####
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


#training with multinomial naive bayes

nb = MultinomialNB()
#fit
nb.fit(X_train,y_train)

####Predictions & Evaluations using confustion matrix###

predictions = nb.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

##precision is 0.92

#Text Processings 

#Creating pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts  
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

pipeline.fit(X_train,y_train)

predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#precision is 0.66 :(  Tf-ldf made things worse
#got preceision to .93 by taking out Tf-ldf


