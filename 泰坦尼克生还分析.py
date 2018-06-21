#决策树
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
X = titanic[['pclass','age','sex']]
y = titanic['survived']
#print(X.info())

#补充数据，age用中位数或者平均数填充
X['age'].fillna(X['age'].mean(),inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=33)
#特征转换
from sklearn.feature_extraction import  DictVectorizer
vec = DictVectorizer(sparse=False)
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
print(vec.feature_names_)

X_test = vec.transform(X_test.to_dict(orient='record'))
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_predict = dtc.predict(X_test)

#性能评测
from sklearn.metrics import classification_report
print(dtc.score(X_test, y_test))
print(classification_report(y_predict,y_test,target_names=['died','survived']))
