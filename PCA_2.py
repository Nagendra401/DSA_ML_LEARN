import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import datasets

iris = datasets.load_iris()
#convert it to numpy arrays
iris_data = iris.data
iris_data
iris_np= scale(iris_data)
print(iris_np[:10])
print(iris_data[:10])

pca = PCA(n_components=4)
pca.fit(iris_np)

#Display the variance defined by each component
var= pca.explained_variance_ratio_
print(var)

#Derive Cumulative variance as percentage
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print (var1)
plt.plot(var1)

#Transformed components
iris_np_pca =  pca.fit_transform(iris_np)
print (iris_np_pca[:10])

#get back original components from Transformed components
iris_pca_original = pca.inverse_transform(iris_np_pca)
print(iris_pca_original[:10])