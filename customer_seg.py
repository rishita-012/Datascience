# Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import scipy.cluster.hierarchy as sch

# Step 2: Load the dataset
df = pd.read_csv(r"C:\Users\rishi\Downloads\archive\Mall_Customers.csv")
print("First 5 rows of data:")
print(df.head())

# Step 3: Select features (Age, Income, Spending Score)
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Elbow Method for Optimal K
wcss = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 4))
plt.plot(range(2, 11), wcss, marker='o')
plt.title("Elbow Method - Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# Step 6: Apply K-Means Clustering (K=5 as an example)
kmeans = KMeans(n_clusters=5, random_state=42)
k_labels = kmeans.fit_predict(X_scaled)

# Evaluation for K-Means
sil_score_kmeans = silhouette_score(X_scaled, k_labels)
db_index_kmeans = davies_bouldin_score(X_scaled, k_labels)

print("K-Means Silhouette Score:", sil_score_kmeans)
print("K-Means Davies–Bouldin Index:", db_index_kmeans)

# Add cluster labels to DataFrame
df['Cluster_KMeans'] = k_labels

# Visualize K-Means Clusters
sns.pairplot(df, hue='Cluster_KMeans', vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
plt.suptitle("K-Means Clustering", y=1.02)
plt.show()

# Step 7: Dendrogram for Hierarchical Clustering
plt.figure(figsize=(10, 5))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title("Dendrogram - Hierarchical Clustering")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# Step 8: Apply Agglomerative (Hierarchical) Clustering (5 clusters)
hc = AgglomerativeClustering(n_clusters=5)
hc_labels = hc.fit_predict(X_scaled)

# Evaluation for Hierarchical Clustering
sil_score_hc = silhouette_score(X_scaled, hc_labels)
db_index_hc = davies_bouldin_score(X_scaled, hc_labels)

print("Hierarchical Silhouette Score:", sil_score_hc)
print("Hierarchical Davies–Bouldin Index:", db_index_hc)

# Add cluster labels to DataFrame
df['Cluster_HC'] = hc_labels

# Visualize Hierarchical Clusters
sns.pairplot(df, hue='Cluster_HC', vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
plt.suptitle("Hierarchical Clustering", y=1.02)
plt.show()
