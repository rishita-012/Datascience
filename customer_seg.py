# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import scipy.cluster.hierarchy as sch

# Title
st.title("Customer Segmentation using Clustering")
st.markdown("Mall Customer Segmentation using **K-Means** and **Hierarchical Clustering**")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/Mall_Customers.csv")
    return df

df = load_data()
st.subheader("First 5 Rows of Data")
st.write(df.head())

# Feature Selection
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Plot
def plot_elbow():
    wcss = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(6,4))
    plt.plot(range(2, 11), wcss, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.grid(True)
    st.pyplot(plt)

st.subheader("Elbow Method for Optimal K")
plot_elbow()

# Clustering Option
algo = st.selectbox("Select Clustering Algorithm", ["K-Means", "Hierarchical"])
n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=5)

if algo == "K-Means":
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_scaled)
    df['Cluster'] = labels
    
    # Evaluation
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    st.success(f"K-Means Silhouette Score: {sil:.3f}")
    st.info(f"K-Means Davies–Bouldin Index: {db:.3f}")

elif algo == "Hierarchical":
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)
    df['Cluster'] = labels
    
    # Evaluation
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    st.success(f"Hierarchical Silhouette Score: {sil:.3f}")
    st.info(f"Hierarchical Davies–Bouldin Index: {db:.3f}")

    # Dendrogram
    st.subheader("Dendrogram")
    plt.figure(figsize=(10, 5))
    dendro = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
    plt.title("Dendrogram")
    plt.xlabel("Customers")
    plt.ylabel("Euclidean Distance")
    st.pyplot(plt)

# Visualize Clusters
st.subheader(f"{algo} Cluster Visualization")
sns.set(style="whitegrid")
pair_plot = sns.pairplot(df, hue='Cluster', vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
st.pyplot(pair_plot)

