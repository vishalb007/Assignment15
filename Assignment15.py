from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets

def wine_classifier():
	wine_data=datasets.load_wine()

	xtrain,xtest,ytrain,ytest=train_test_split(wine_data.data,wine_data.target,test_size=0.3)

	model=KNeighborsClassifier(n_neighbors=3)
	model.fit(xtrain,ytrain)
	predict=model.predict(xtest)

	print("Accuracy is : ",accuracy_score(ytest,predict))

def main():
	wine_classifier()

if __name__=="__main__":
	main()