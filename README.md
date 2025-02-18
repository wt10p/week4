Satellite Echo Classification
This repository uses machine learning to classify satellite echoes into sea ice and leads. It generates average echo shapes, computes standard deviations, and compares results with ESA's official classification using a confusion matrix.

Key Features
Colocation: Combines Sentinel-3 OLCI/SRAL and Sentinel-2 optical data.

Unsupervised Learning: Uses K-means and Gaussian Mixture Models (GMM) for classification.

Evaluation: Compares results with ESA data using confusion matrices.

Steps
1. Colocating Sentinel-3 and Sentinel-2 Data
Retrieve metadata for Sentinel-2 and Sentinel-3 OLCI/SRAL.

Co-locate data based on geospatial footprints and timestamps.

Visualize overlapping satellite observations.

2. Unsupervised Learning
K-means Clustering:

Groups data into clusters based on similarity.

Efficient for large datasets but requires predefined cluster count.

Gaussian Mixture Models (GMM):

Probabilistic clustering using Gaussian distributions.

Handles overlapping clusters and varying shapes.

3. Image Classification
Apply K-means and GMM to Sentinel-2 optical data.

Classify sea ice and leads based on spectral reflectance.

4. Altimetry Classification
Process Sentinel-3 SAR altimetry data.

Use waveform characteristics (e.g., peakiness, stack standard deviation) for classification.

Visualize clustered data and waveform alignment.

5. Comparison with ESA Data
Compare GMM results with ESA's official classification.

Generate a confusion matrix and classification report.

Code Snippets
K-means Clustering

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# K-means model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
Gaussian Mixture Model (GMM)

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# GMM model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.show()
Confusion Matrix and Classification Report

from sklearn.metrics import confusion_matrix, classification_report

# True labels and predicted labels
true_labels = flag_cleaned_modified
predicted_gmm = clusters_gmm

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predicted_gmm))

# Classification Report
print("Classification Report:")
print(classification_report(true_labels, predicted_gmm))
Results
K-means: Effective for basic clustering but limited by spherical cluster assumptions.

GMM: Handles overlapping clusters and varying shapes, providing better accuracy.

Comparison with ESA Data: High accuracy (100%) in distinguishing sea ice and leads.

Dependencies
Python Libraries: NumPy, Pandas, Matplotlib, Scikit-learn, Rasterio, netCDF4.

Data Sources: Sentinel-2 and Sentinel-3 satellite data.

This simplified version focuses on the core functionality and key steps, making it easier to understand and implement. For detailed code and visualizations, refer to the original repository. 🚀
