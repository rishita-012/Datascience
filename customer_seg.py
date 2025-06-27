# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import scipy.cluster.hierarchy as sch

# App Title
st.title("🛍️ Mall Customer Segmentation")
st.markdown("Upload your dataset and perform **K-Means** and **Hierarchical Clustering**.")

# Upload CSV
uploaded_file = st.file_uploader("Upload Mall_Customers.csv", type="csv")

# Main workflow
if uploaded_file is not None:
    # Read the CSV
    df = pd.read_csv(uploaded_file)

    # Show data
    st.subheader("First 5 Rows of Dataset")
    st.write(df.head())

    # Feature selection
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Plot
    st.subheader("Elbow Method for K-Means")
    wcss = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(range(2, 11), wcss, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    st.pyplot(plt)

    # Select clustering algorithm
    algo = st.selectbox("Choose Clustering Algorithm", ["K-Means", "Hierarchical"])
    n_clusters = st.slider("Select Number of Clusters", 2, 10, value=5)

    if algo == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
        df['Cluster'] = labels

        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        st.success(f"K-Means Silhouette Score: {sil:.3f}")
        st.info(f"K-Means Davies–Bouldin Index: {db:.3f}")

    elif algo == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
        df['Cluster'] = labels

        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        st.success(f"Hierarchical Silhouette Score: {sil:.3f}")
        st.info(f"Hierarchical Davies–Bouldin Index: {db:.3f}")

        # Dendrogram
        st.subheader("Dendrogram")
        plt.figure(figsize=(10, 5))
        sch.dendrogram(sch.linkage(X_scaled, method='ward'))
        plt.title("Dendrogram")
        plt.xlabel("Customers")
        plt.ylabel("Euclidean Distance")
        st.pyplot(plt)

    # Show cluster visualization
    st.subheader("Cluster Visualization")
    sns.set(style="whitegrid")
    pair_plot = sns.pairplot(df, hue='Cluster', vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
    st.pyplot(pair_plot)

else:
    st.warning("📂 Please upload the Mall_Customers.csv file to proceed.")
