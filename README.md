# GEOL0069_Satellite Echo Classification
This repository uses machine learning to classify echoes into **sea ice** and **leads**. It calculates average echo shapes, computes standard deviations, and evaluates results against ESA's classification using a confusion matrix.


This project combines Sentinel-3 (OLCI & SRAL) and Sentinel-2 optical data, using unsupervised learning to classify **sea ice** and **leads**. It aims to create an automated pipeline for enhanced Earth Observation (EO) analysis by integrating satellite datasets and applying machine learning to classify environmental features.

Goals：
Learn to colocate multi-sensor satellite data.

Apply unsupervised learning to classify environmental features like sea ice and leads.

Combine altimetry and optical data for enhanced classification.

Evaluate results using confusion matrices and accuracy metrics.

Satellite data analysis is a dynamic field, and methods may vary by dataset. This project offers a practical guide to integrating remote sensing and machine learning.



This project employs a variety of Python libraries and geospatial tools to process, analyze, and classify Earth Observation (EO) data. The key dependencies include:

Scikit-Learn – Utilized for implementing machine learning models such as K-Means and GMM.

Rasterio – For handling Sentinel-2 geospatial raster data.

Shapely – Used for performing geometric operations, particularly in colocation analysis.

netCDF4 – Enables the processing of Sentinel-3 altimetry data.

Folium – Facilitates the visualization of geospatial data.

NumPy – Numerical computations and matrix operations.

Requests – Handles API calls for retrieving Sentinel-3 metadata.

Pandas – Manages data manipulation and tabular processing.

Matplotlib – Visualizes classification results effectively.

These tools collectively enable comprehensive analysis and visualization of EO data


Colocating Sentinel-3 OLCI/SRAL and Sentinel-2 Optical Data
This section focuses on combining Sentinel-2 and Sentinel-3 data to enhance Earth observation analysis. Sentinel-2 provides high spatial resolution, while Sentinel-3 offers broader coverage and altimetry insights, improving environmental monitoring, especially for sea ice and lead classification. Below are the steps to align and analyze these datasets effectively.

Step 0: Load Essential Functions
The process starts by loading necessary functions to retrieve Sentinel-2 and Sentinel-3 metadata. Google Drive is mounted in Google Colab for file access. Using libraries like requests, pandas, shapely, and folium, the script fetches, processes, and visualizes data from the Copernicus Data Space Ecosystem. Authentication is handled via access tokens, and data is queried by date, location, and cloud cover. Sentinel-3 OLCI, SRAL, and Sentinel-2 optical data are retrieved, with footprints processed for geographic overlap. Results are visualized on interactive maps, and timestamps are standardized for accurate analysis.

Step 1: Retrieve Metadata for Sentinel-2 and Sentinel-3 OLCI
Metadata for Sentinel-2 and Sentinel-3 OLCI is retrieved separately to identify common observation locations. Authentication is required to obtain access tokens. The script queries Sentinel-3 OLCI and Sentinel-2 data using specific functions, applying a 0–10% cloud cover filter for Sentinel-2. Metadata is saved as CSV files (sentinel3_olci_metadata.csv and sentinel2_metadata.csv) for alignment. Key details like product IDs, acquisition times, and footprints are displayed in structured tables for inspection.

Co-locate the Data
Co-location pairs are identified by matching Sentinel-2 and Sentinel-3 OLCI data based on their geo_footprint. Timestamps are standardized for consistent comparisons. The check_collocation() function detects overlapping observations within a 10-minute window, and results are stored in a DataFrame. The first five co-located observations are visualized on an interactive map using folium.

Step 2: Download Sentinel-3 OLCI Data
The next step involves downloading Sentinel-3 OLCI data using a structured approach. The download_single_product() function retrieves the data from Copernicus Dataspace, storing it in a specified directory. Users can customize the download by modifying the product_id, file_name, and download_dir.

Step 3: Integrate Sentinel-3 SRAL Data
Sentinel-3 SRAL altimetry data is integrated into the analysis. The query_sentinel3_sral_arctic_data() function retrieves SRAL metadata for a specified date range, saved as s3_sral_metadata.csv. Metadata files are loaded, and timestamps are standardized. The check_collocation() function identifies overlapping Sentinel-2 and Sentinel-3 SRAL observations within a 10-minute window. Results are visualized on an interactive map, showing the geographical footprints of overlapping data.


## Unsupervised Learning

This section explores unsupervised learning in Earth Observation (EO), focusing on classification tasks to detect patterns and group data without predefined labels. Key tasks include distinguishing sea ice from leads using Sentinel-2 optical data and classifying sea ice and leads using Sentinel-3 altimetry data. By the end, you'll gain a foundation in unsupervised learning for remote sensing and EO analysis.

### Introduction to Unsupervised Learning Methods
Unsupervised learning identifies patterns in data without predefined labels, making it ideal for exploratory analysis and pattern detection in EO.

### Introduction to K-means Clustering
K-means clustering is an unsupervised algorithm that partitions data into **k clusters** based on feature similarity. The process involves:
1. Initializing **k centroids**.
2. Assigning data points to the nearest centroid.
3. Updating centroids based on cluster means.
4. Iterating until centroids stabilize.

K-means is widely used for pattern recognition, data segmentation, and exploratory analysis.

### Reasons of using K-means for Clustering
- **No prior knowledge required**: Ideal for unknown data structures.
- **Efficient and scalable**: Handles large datasets with minimal complexity.
- **Versatile**: Suitable for real-world applications like EO analysis.

### Key Components of K-means
1. **Choosing k**: The number of clusters must be predefined and impacts results.
2. **Centroid initialization**: Affects final clustering outcomes.
3. **Assignment step**: Data points are grouped by proximity to the nearest centroid using squared Euclidean distance.
4. **Update step**: Centroids are recalculated based on the mean position of assigned points.

### The Iterative Process of K-means
K-means iterates through assignment and update steps until centroids stabilize, minimizing intra-cluster variation. While it converges to an optimal solution, it may sometimes settle on a local optimum.

### Advantages of K-means
- **Efficiency**: Works well with large datasets.
- **Interpretability**: Provides clear insights into data patterns.

### Basic Code Implementation
Below is a practical implementation of K-means clustering:
1. **Setup**: Mount Google Drive and install necessary libraries (`Rasterio` for geospatial data, `netCDF4` for scientific data).
2. **Data Generation**: Create 100 random data points.
3. **Model Initialization**: Initialize a K-means model with 4 clusters.
4. **Clustering**: Assign data points to clusters using `kmeans.fit(X)`.
5. **Visualization**: Plot clusters with color-coded points and mark centroids with black dots.
6. # Python code for K-means clustering
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

```python
# Python code for K-means clustering
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
```


![image](https://github.com/user-attachments/assets/34a3d752-b673-4daa-91c9-59705b227234)


### Gaussian Mixture Models (GMM)

#### Introduction
Gaussian Mixture Models (GMM) are a probabilistic clustering technique that models data as a combination of multiple Gaussian distributions, each with its own mean and variance. GMMs are widely used for clustering and density estimation, offering flexibility in representing complex data distributions.

#### Reasons why Use GMM for Clustering?
- **Soft Clustering**: Assigns probabilities to data points for belonging to multiple clusters, capturing uncertainty and overlapping structures.
- **Flexible Cluster Shapes**: Unlike K-means, GMM adapts to varying cluster shapes and sizes by adjusting the covariance structure of each Gaussian component.
- **Effective for Complex Data**: Ideal for datasets with overlapping distributions or varying densities.

#### Key Components of GMM
1. **Number of Components**: Determines how many Gaussian distributions will model the data.
2. **Expectation-Maximization (EM) Algorithm**: 
   - **E-step**: Estimates the probability of each data point belonging to a Gaussian component.
   - **M-step**: Updates the mean, variance, and weight of each Gaussian to maximize likelihood.
3. **Covariance Structure**: Allows for flexible cluster shapes (spherical, diagonal, tied, or fully adaptable).

#### The EM Algorithm in GMM
The EM algorithm iteratively optimizes clustering:
1. **E-step**: Assigns probabilities to data points for each Gaussian component.
2. **M-step**: Updates Gaussian parameters (mean, variance, weight) to maximize likelihood.
3. **Convergence**: Repeats until parameters stabilize, ensuring an optimal fit.

#### Advantages of GMM
- **Probabilistic Clustering**: Captures uncertainty and overlapping structures.
- **Flexible Cluster Shapes**: Adapts to varying sizes, orientations, and densities.
- **Effective for Complex Data**: Handles datasets with overlapping distributions.

#### Basic Code Implementation
Below is a simple GMM implementation using `sklearn.mixture` and `matplotlib`:
1. **Setup**: Import necessary libraries (`GaussianMixture`, `matplotlib`, `numpy`).
2. **Data Generation**: Create 100 random 2D data points.
3. **Model Initialization**: Initialize a GMM with 3 components.
4. **Clustering**: Fit the model to the data and predict cluster assignments.
5. **Visualization**: Plot data points color-coded by cluster, with cluster centers highlighted in black.


