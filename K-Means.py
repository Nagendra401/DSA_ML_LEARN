import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt

#Load the digits dataset
digits = datasets.load_digits()

print(digits.DESCR)

#Display the first digit
plt.figure(1, figsize=(8, 8))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

digits_data,digits_target = digits.data[:-1],digits.target[:-1]
digits_target[1]
digits_target[2]
digits_target[3]
digits_data[0]
# Number of clusters
kmeans = KMeans(n_clusters=5)
# Fitting the input data
kmeans = kmeans.fit(digits_data)
# Getting the cluster labels
groups = kmeans.predict(digits_data)
# Centroid values
centroids = kmeans.cluster_centers_

print(centroids) # From sci-kit learn
print(len(groups))
print(groups[:5])

temp_labels = pd.DataFrame(groups)
temp_actuals= pd.DataFrame(digits_target)
temp_actuals.head(20)

df_c = pd.concat([temp_labels, temp_actuals],axis=1)
df_c.columns = ['Groups','Labels']
df_c.head(10)
df_sorted = df_c.sort_values( by =['Groups', 'Labels'], ascending = [True, True])
df_sorted.head(10)
