#Naive Bayes
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset = 'all')
# print(len(news.data))
# print(news.data[0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)

from sklearn.feature_extraction.text import CountVectorizer
vec=CountVectorizer()
X_train=vec.fit_transform(X_train)
X_test=vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(X_train,y_train)
y_predict=mnb.predict(X_test)

#评估
from sklearn.metrics import classification_report
print("The accuracy of Naive Bayes Classifier is", mnb.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=news.target_names))
