import sklearn
import pandas
from sklearn import datasets,model_selection,svm,metrics
from sklearn.neighbors import KNeighborsClassifier
import pickle
cancer = sklearn.datasets.load_breast_cancer()
X=cancer.data
y=cancer.target
xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(X,y,test_size=0.2)
model=svm.SVC(kernel='poly',degree=3)  #poly,sigmoid,linear
#model=KNeighborsClassifier(n_neighbors=5)
model.fit(xtrain,ytrain)
#with open('SVM.pickle','wb') as f:
#    pickle.dump(model,f)
#svmfile=open('SVM.pickle','rb')
#model=pickle.load(svmfile)
prediction=model.predict(xtest)
acc=metrics.accuracy_score(prediction,ytest)   #for svm
print(acc)
#for i in range(len(xtest)):
#    print(prediction[i],ytest[i])
