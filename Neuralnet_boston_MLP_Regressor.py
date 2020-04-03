from sklearn import datasets
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math

boston = datasets.load_boston()
boston
bost_data = boston.data
bost_target = boston.target
len(bost_data)
len(bost_target)

#Data Preprocessing
# apply standarization to the feature values
#mimax method used to scale data in interval [0,1]
#scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(bost_data)
# Now apply the transformations to the data:
bost_data = scaler.transform(bost_data)
len(bost_data)
bost_data[0]

scaler.fit(bost_target.reshape(-1,1))
bost_target = scaler.transform(bost_target.reshape(-1,1))
len(bost_target)
bost_target[0]
################ Training the model #################
#---------------- Regression begins -----------------
#split data to train and test
#from sklearn.cross_validation import train_test_split
#train_test_split divide test 25%, train 75% data
X_train, X_test, y_train, y_test = train_test_split(bost_data, bost_target, test_size=0.25, random_state=133)
print ("Shape of (X_train, X_test, Y_train, Y_test)")
print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#MLPClassifier
#Multi-Layer Perceptron Regressor
"""hidden_layer_sizes. For this parameter you pass in a tuple consisting of the number of neurons you want at each layer,
where the nth entry in the tuple represents the number of neurons in the nth layer of the MLP model"""
clf = MLPRegressor(hidden_layer_sizes=(5,3), learning_rate='constant', learning_rate_init=0.01, max_iter=200000, tol= 0.03, activation='logistic',solver='sgd',verbose=True)
clf.fit(X_train,y_train.ravel())

##################### Predictions and Evaluation ################
predictions = clf.predict(X_test)
math.sqrt(mean_squared_error(y_test,predictions.reshape(-1,1)))

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

#Unscale the data
unscale = scaler.inverse_transform(predictions.reshape(-1,1))
