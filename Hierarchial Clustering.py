from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import datasets
from scipy.cluster.hierarchy import fcluster

iris = datasets.load_iris()
print(iris.feature_names)
#convert it to numpy arrays
iris_data = iris.data
print(iris_data)
len(iris_data)
#simple, complete or average linkages can be used
Z = linkage(iris_data,'average')


#array has the format [idx1, idx2, dist, sample_count]
"""All indices idx >= len(iris_data) actually refer to the cluster formed in Z[idx - len(iris_data)].
This means that while idx 149 corresponds to iris_data[149] that idx 150 corresponds to the cluster formed in Z[0], 
idx 151 to Z[1], 152 to Z[2]"""
len(Z)
print(Z[0])
print(Z[1])
print(Z[21])
print(Z[150])
#Print basic dendogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8,  # font size for the x axis labels
)
plt.show()

#Print the truncated dendogram
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
plt.axhline(y = 1.9, c = 'black')

dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    leaf_rotation=90., #rotates the axis labels
    leaf_font_size=12, #font size for the x axis labels
    show_contracted=True,  # to get a distribution impression in truncated branches
    )
plt.show()



#If we decide to draw a cutoff at Max_d
max_d = 1.9
clusters = fcluster(Z, max_d, criterion='distance')
print(clusters)


plt.figure(figsize=(10, 8))
plt.scatter(iris_data[:,2], iris_data[:,3], c=clusters, cmap='prism')  # plot points with cluster dependent colors
plt.show()

#If we observe K clusters
k=3
cluster2 = fcluster(Z, k, criterion='maxclust')
print(cluster2)
plt.figure(figsize=(10, 8))
plt.scatter(iris_data[:,2], iris_data[:,3], c=cluster2, cmap='prism')  # plot points with cluster dependent colors
plt.show()

