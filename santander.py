## CS 484 - Project
## March 2021
## Anh Nguyen & Jake Zegeer

#warnings import used to prevent unwanted exceptions while running code
#we used this because we imported numerous packages which pandas does not
#react well, causing irrelevant warnings.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#pandas
import pandas as pd
#matplotlib to plot graphs/visual reflections
import matplotlib.pyplot as plt
#python arrays/matrix numpy
import numpy as np
#seaborn to plot graphs/visual reflections
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import scikitplot as skplt

class Customer_Prediction:

	def __init__(self):
		## Read train data to a table
		train_data = pd.read_csv('train.csv')

		## Read test data to a table
		test_data = pd.read_csv('test.csv')
	
	def read_train_data(self):
		## Read train data to a table
		train_data = pd.read_csv('train.csv')
		
		target = train_data['target']
		sns.set_style('whitegrid')
		sns.countplot(train_data.target)
		plt.savefig('target.png')
		
		#Rid of the string labels so we can deal with numerical data only
		data=train_data.drop(['ID_code','target'],axis=1)

		#changes data only of float16 type
		num = ['float16']
		train_data.select_dtypes(include=num)

		#Rid of attribute label names and onlt deal with numerical data
		train_data.drop('ID_code',axis=1)
		train_data.drop('target',axis=1)
		
		return train_data

	def read_test_data(self):
		## Read test data to a table
		test_data = pd.read_csv('test.csv')

		return test_data

	def target_plot(self,train_data,test_data):

		label_list = [x for x in train_data.columns if x not in ['ID_code', 'target']]
		#working version of plot of mean value distribution of rows for both train and test data
		print(train_data.head())
		plt.figure(figsize=(10,4))
		#computes standard deviation for both train and test data
		sns.distplot(train_data[label_list].mean(axis=1), color="green", label='training data')
		sns.distplot(test_data[label_list].mean(axis=1), color="pink", label='test data')
		plt.legend()
		plt.savefig('Distribution.png')
	
		

	def TSNE(self, train_data):

		target = train_data['target']

		train_data=train_data.drop(['ID_code','target'],axis=1)
		

		reduction = MinMaxScaler()
		train_data2 = reduction.fit_transform(train_data)
		print(train_data2.shape)

	
		#TSNE algorithm:

		#takes two categories
		model=TSNE(n_components=2)
		#TSNE computation
		data=model.fit_transform(train_data2)
		#The transpose is taken to help estimate the covariances in regression
		data=np.vstack((data.transpose(),target)).transpose()

		finished=pd.DataFrame(data=data, columns=("X-Axis", "Y-Axis", "Transactions"))
		
		sns.FacetGrid(finished,hue="Transactions", size=6).map(plt.scatter, "X-Axis", "Y-Axis").add_legend()
		plt.title("TSNE Visual")
		#plt.show()
		plt.tight_layout()
		plt.savefig('TSNE_Visual.png')

	def treeForest(self,train_data):

		#Rid of the string labels so we can deal with numerical data only
		data=train_data.drop(['ID_code','target'],axis=1)
		target = train_data['target']

		X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.20,stratify=target)

		#Random Forest decision tree algorithm
		tree=RandomForestClassifier()
		tree.fit(X_train,y_train)
		
		#cross-validation used to estimate training data along test data
		tree = CalibratedClassifierCV(tree, method="sigmoid")
		tree.fit(X_train, y_train)

		
		#plot figure of randomforest with the cross-valiation active
		plt.figure(figsize=(10,4))
		#probabiloty of '1' occurance in training data target label
		predict_y = tree.predict_proba(X_train)
		#probabiloty of '1' occurance in test data target label
		predict_y = tree.predict_proba(X_test)
		#plots the ROC Curve 
		skplt.metrics.plot_roc_curve(y_test, predict_y)
		plt.title('Receiver Operating Characteristic Curve on Train data against test data')
		plt.savefig('Random_forest.png')
		print('ROC Score for training on test data :',roc_auc_score(y_test,predict_y[:,1]))

	def logReg(self,train_data):

		#Rid of the string labels so we can deal with numerical data only
		data=train_data.drop(['ID_code','target'],axis=1)
		target = train_data['target']

		X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.20,stratify=target)

		#Random Forest decision tree algorithm
		tree=LogisticRegression()
		tree.fit(X_train,y_train)
		
		#cross-validation used to estimate training data along test data
		tree = CalibratedClassifierCV(tree, method="sigmoid")
		tree.fit(X_train, y_train)

		
		#plot figure of randomforest with the cross-valiation active
		plt.figure(figsize=(10,4))
		#probabiloty of '1' occurance in training data target label
		predict_y = tree.predict_proba(X_train)
		#probabiloty of '1' occurance in test data target label
		predict_y = tree.predict_proba(X_test)
		#plots the ROC Curve 
		skplt.metrics.plot_roc_curve(y_test, predict_y)
		plt.title('Receiver Operating Characteristic Curve on Train data against test data')
		plt.savefig('Log_Regression.png')
		print('ROC Score for training on test data :',roc_auc_score(y_test,predict_y[:,1]))

	def gnb(self,train_data):

		#Rid of the string labels so we can deal with numerical data only
		data=train_data.drop(['ID_code','target'],axis=1)
		target = train_data['target']

		X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.20,stratify=target)

		#Random Forest decision tree algorithm
		tree=GaussianNB()
		tree.fit(X_train,y_train)
		
		#cross-validation used to estimate training data along test data
		tree = CalibratedClassifierCV(tree, method="sigmoid")
		tree.fit(X_train, y_train)

		
		#plot figure of randomforest with the cross-valiation active
		plt.figure(figsize=(10,4))
		#probabiloty of '1' occurance in training data target label
		predict_y = tree.predict_proba(X_train)
		#probabiloty of '1' occurance in test data target label
		predict_y = tree.predict_proba(X_test)
		#plots the ROC Curve 
		skplt.metrics.plot_roc_curve(y_test, predict_y)
		plt.title('Receiver Operating Characteristic Curve on Train data against test data')
		plt.savefig('GNB.png')
		print('ROC Score for training on test data :',roc_auc_score(y_test,predict_y[:,1]))

if __name__ == "__main__":
    obj = Customer_Prediction()
    train_data=obj.read_train_data()
    #test_data=obj.read_test_data()
    #obj.target_plot(train_data,test_data)
    #obj.TSNE(train_data)
    obj.treeForest(train_data)
    obj.Log_Regression(train_data)
    obj.gnb(train_data)
	


















