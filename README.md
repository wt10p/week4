# GEOL0069_Unsupervised_Learning_Methods
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

```python
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
```
![image](https://github.com/user-attachments/assets/26351cee-c6d9-48b1-b36d-cfa89f5bcd89)

Image Classification
This section uses unsupervised learning for image classification, focusing on distinguishing sea ice from leads in Sentinel-2 imagery. By applying clustering algorithms, patterns can be identified and classified without labeled data, enhancing the analysis of remote sensing data.

K-Means Implementation
This Python script applies K-means clustering to a Sentinel-2 satellite image, classifying surface features based on reflectance values from the Red (B4) band. Here’s a simplified breakdown:

Libraries:

rasterio: For loading satellite imagery.

numpy: For numerical operations.

KMeans from sklearn.cluster: For clustering.

matplotlib.pyplot: For visualization.

Steps:

Load Band 4 using rasterio.open() and store pixel values in a NumPy array.

Apply a valid data mask to filter out no-data pixels.

Reshape the image data into a 1D array.

Apply K-means with 2 clusters (n_clusters=2) to classify surface types (e.g., sea ice and open water).

Reshape the classified labels to match the original image dimensions, assigning -1 to masked pixels.

Visualize the results using plt.imshow(), with distinct colors representing different clusters.

This approach enables unsupervised classification of Sentinel-2 imagery, providing insights into surface variations for applications like climate monitoring, environmental research, and land cover classification—without requiring labeled training data.

```python
import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

base_path = "/content/drive/MyDrive/GEOL0069/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4', 'B3', 'B2']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for K-means, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 3))

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place cluster labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('K-means clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()
```

![image](https://github.com/user-attachments/assets/5f7c5350-fad0-49e4-bd55-ad49512ccef9)

## GMM Implementation

This Python script applies Gaussian Mixture Model (GMM) clustering to a Sentinel-2 B4 optical band using `scikit-learn`. Here’s a simplified breakdown:

1. **Libraries**:
   - `rasterio`: For reading the B4 band.
   - `numpy`: For data manipulation.
   - `GaussianMixture` from `sklearn.mixture`: For GMM clustering.
   - `matplotlib.pyplot`: For visualization.

2. **Steps**:
   - Read the B4 band using `rasterio` and store the data in a NumPy array.
   - Create a mask to exclude zero-value (no-data) pixels.
   - Reshape the valid data for clustering.
   - Apply GMM with 2 components to perform soft clustering, where each pixel is assigned a probability of belonging to each cluster.
   - Store the results in an array, assigning -1 to no-data areas.
   - Visualize the clusters using `matplotlib`, with distinct colors representing different clusters.

This method is effective for analyzing sea ice, land cover, and environmental patterns in Sentinel-2 imagery, providing a probabilistic approach to unsupervised classification.

```python
import rasterio
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Paths to the band images
base_path = "/content/drive/MyDrive/GEOL0069/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4', 'B3', 'B2']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for GMM, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 3))

# GMM clustering
gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
labels = gmm.predict(X)

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place GMM labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('GMM clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()
```
![image](https://github.com/user-attachments/assets/28386ef2-8b7c-49ee-b6ca-15d024bcc14f)



