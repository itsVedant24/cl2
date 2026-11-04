# pip install pandas numpy matplotlib scipy scikit-learn


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load Iris dataset
iris = load_iris()
X = iris.data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create dendrogram
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# Apply Agglomerative Clustering
model = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

labels = model.fit_predict(X_scaled)

# Display cluster results
print("Agglomerative Hierarchical Clustering")
print("--------------------------------------")

for i in range(3):
    print("\nCluster", i + 1)
    print("Number of data points:", list(labels).count(i))

print("\nFirst 10 Cluster Labels:")
print(labels[:10])

# Visualize clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels
)

plt.title("Agglomerative Hierarchical Clustering")
plt.xlabel("Sepal Length (Standardized)")
plt.ylabel("Sepal Width (Standardized)")
plt.show()



# 4. Agglomerative Hierarchical Clustering
# Agglomerative Hierarchical Clustering is an unsupervised machine 
# learning algorithm that groups similar data points into clusters. 
# It follows a bottom-up approach in which every data point initially 
# forms a separate cluster. The closest clusters are gradually merged 
# until the required number of clusters is obtained. In this practical, 
# the Iris dataset is used, and a dendrogram is generated to represent 
# the clustering process.