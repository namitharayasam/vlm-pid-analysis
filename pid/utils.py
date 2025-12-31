# vlm-pid-analysis/pid/utils.py

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, MiniBatchKMeans

def cluster_embeddings(embeddings, target_samples):
    """
    Reduces the number of samples/sequence length using MiniBatchKMeans
    to cluster points and return cluster centers.
    """
    num_samples = embeddings.shape[0]

    if num_samples <= target_samples:
        return embeddings

    # Convert to numpy for sklearn
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach().numpy()
        
    kmeans = MiniBatchKMeans(n_clusters=target_samples, random_state=42, 
                             batch_size=256, n_init=1, max_iter=50)
    kmeans.fit(embeddings)

    # Return cluster centers (the compressed/aligned representation)
    return kmeans.cluster_centers_

def clustering(X, pca=False, n_clusters=10, n_components=10):
    """
    Performs clustering (with optional PCA) to get discrete labels for an input variable.
    Returns: discrete labels (kmeans.labels_) and the processed data (X).
    """
    # Convert to numpy and handle NaNs
    if isinstance(X, torch.Tensor):
        X = X.cpu().detach().numpy()
        
    X = np.nan_to_num(X)
    
    # Reshape if the embedding is higher than 2D (e.g., from feature maps)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
        
    # Apply PCA for dimensionality reduction if requested
    if pca:
        X = normalize(X)
        n_components_actual = min(n_components, X.shape[1], X.shape[0])
        if n_components_actual > 0:
            X = PCA(n_components=n_components_actual, random_state=42).fit_transform(X)

    # Ensure n_clusters does not exceed the number of samples
    n_clusters_actual = min(n_clusters, X.shape[0])

    kmeans = KMeans(n_clusters=n_clusters_actual, n_init=1, max_iter=50, random_state=42).fit(X)
    
    return kmeans.labels_, X