from sklearn.datasets import load_iris
iris = load_iris()
#print(iris.data.shape)
#print(iris.DESCR)

#随机采样
from sklearn.model_selection import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)

#KNN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

ss = StandardScaler()
X_train=ss.fit_transform(X_train)
X_test = ss.transform(X_test)

knc=KNeighborsClassifier()
knc.fit(X_train,y_train)
y_predict= knc.predict((X_test))

#评估
print("The accuracy of K-nearest Neighbor Classifier is", knc.score(X_test,y_test))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=iris.target_names))