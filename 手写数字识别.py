#SVM
from sklearn.datasets import load_digits
digits = load_digits()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,test_size=0.25,random_state=33)

#识别
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lsvc = LinearSVC()
lsvc.fit(X_train,y_train)
y_predict = lsvc.predict(X_test)

#评估
print('The Accuracy of Linear SVC is', lsvc.score(X_test,y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))