from urllib import request
url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv'
request.urlretrieve(url,'cell-samples.csv')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
df=pd.read_csv('cell-samples.csv')
df.columns
df.head()
df.tail()

#to check how many classes there are
df['Class'].value_counts()
#there are two classes 2(benign) and 4(malignant)

#plot UnifSize vs Clump and mark the cells as benign and malignant
class2=df.loc[df['Class'] == 2]
class2
class4=df.loc[df['Class'] == 4]
class4
plt.scatter(class2['Clump'],class2['UnifSize'],color='blue',label='benign cell')
plt.scatter(class4['Clump'],class4['UnifSize'],color='red',label='cancerous cell')
plt.xlabel('Clump')
plt.ylabel('UnifSize')
plt.legend()
plt.savefig('cells.png',dpi=300)
#plt.show()

#cleaning the data
#checking if all have int values

df.dtypes
#data type of BareNuc is not int
df['BareNuc']
#remove those rows

df=df[pd.to_numeric(df['BareNuc'],errors='coerce').notnull()] 
df['BareNuc']
#notice the change in the number of length

#now change the dtype
df['BareNuc'] = df['BareNuc'].astype('int')
df.dtypes

x=np.asanyarray(df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize','BareNuc', 'BlandChrom', 'NormNucl', 'Mit']])
x[0:5]

y=np.asanyarray(df['Class']) #note one [] and not [[]]. previous one will give flattened values.
y[0:5]

#train/test split

from sklearn.model_selection import train_test_split
train_x, test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=4)

print('train-set:',train_x.shape,train_y.shape)
print('test-set:',test_x.shape,test_y.shape)

#svm

from sklearn import svm
clf=svm.SVC(kernel='rbf') #linear, sigmoid, polynomial
clf.fit(train_x,train_y)

pred_y=clf.predict(test_x)


#accuracy
from sklearn.metrics import jaccard_score
print('jaccard_score:',jaccard_score(test_y, pred_y,pos_label=2))


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
cm= confusion_matrix(test_y, pred_y)
print('confusion matrix=',cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.text(-0.3,-0.3,'true positives')
plt.text(0.7,-0.3,'false positives')
plt.text(-0.3,0.7,'false negatives')
plt.text(0.7,0.7,'true negatives')
plt.savefig('confusion-matrix.png',dpi=300)













