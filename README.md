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


<!-- Unsupervised Learning -->
## Unsupervised Learning
This section applies unsupervised learning to Earth Observation (EO), focusing on classification tasks to detect patterns and group data without predefined labels. The key tasks include distinguishing sea ice from leads using Sentinel-2 optical data and classifying sea ice and leads using Sentinel-3 altimetry data. By the end, you'll have a solid foundation in unsupervised learning for remote sensing and EO analysis.

<!-- Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006] -->
### Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006]
#### Introduction to K-means Clustering
K-means clustering is an unsupervised learning algorithm that partitions data into k clusters based on feature similarity [MacQueen et al., 1967]. The process involves initializing k centroids, assigning data points to the nearest centroid, updating centroids based on cluster means, and iterating until stability is reached. K-means is widely used for pattern recognition, data segmentation, and exploratory analysis, making it a fundamental tool in unsupervised learning.
#### Why K-means for Clustering?
K-means clustering is effective when the data structure is unknown, as it does not require prior knowledge of distribution, making it ideal for exploratory analysis and pattern detection. It is also efficient and scalable, handling large datasets with minimal complexity, making it a preferred choice for real-world applications.

#### Key Components of K-means
K-means clustering relies on key factors: choosing k, which must be predefined and impacts results; centroid initialization, which affects final clustering; the assignment step, where data points are grouped by proximity to the nearest centroid using squared Euclidean distance; and the update step, where centroids are recalculated based on the mean position of assigned points.
#### The Iterative Process of K-means
K-means iterates through assignment and update steps until centroids stabilize, minimizing intra-cluster variation. This ensures convergence to an optimal clustering solution, though it may sometimes settle on a local optimum.
#### Advantages of K-means
K-means is highly efficient, making it ideal for large datasets, and offers easy interpretation, allowing for clear analysis of data patterns.
#### Basic Code Implementation
This section provides a K-means clustering implementation as a practical introduction to the algorithm. In Google Colab, the script mounts Google Drive using drive.mount('/content/drive') for seamless dataset access. It also installs Rasterio for geospatial raster data and netCDF4 for handling large-scale scientific data. Using scikit-learn, the script generates 100 random data points, initializes a K-means model with four clusters, and assigns each point using kmeans.fit(X). A scatter plot visualizes the clusters with color-coded points, while computed centroids are marked with black dots. The plot, displayed with plt.show(), illustrates how K-means groups data for pattern recognition and segmentation.

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

![image](https://github.com/user-attachments/assets/e336776c-92d6-4d6a-b3fc-be0c1f41960d)

Visualization of K-means clustering results on a randomly generated dataset. The colored points represent individual data samples grouped into four clusters, while the black dots indicate the centroids of each cluster, calculated by the K-means algorithm.

<!-- Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006] -->
### Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006]
#### Introduction to Gaussian Mixture Models
Gaussian Mixture Models (GMM) are a probabilistic clustering technique that models data as a combination of multiple Gaussian distributions, each with its own mean and variance [Reynolds et al., 2009]. GMMs are widely used for clustering and density estimation, providing a flexible way to represent complex data distributions.

#### Why Gaussian Mixture Models for Clustering?
Gaussian Mixture Models (GMM) offer significant advantages in clustering by providing a soft clustering approach, where each data point is assigned a probability of belonging to multiple clusters rather than being placed into a single category like in K-means. This probabilistic classification allows for a more nuanced and flexible clustering method, especially when dealing with uncertainty. Additionally, unlike K-means, which assumes clusters are spherical, GMM adapts to varying cluster shapes and sizes by adjusting the covariance structure of each Gaussian component. This makes it particularly effective for datasets with overlapping distributions or varying density regions, providing a more precise and adaptable clustering solution.

#### Key Components of GMM
Gaussian Mixture Models (GMM) require defining the number of components, similar to selecting clusters in K-means, as it determines how many Gaussian distributions will model the data. The model is refined using the Expectation-Maximization (EM) algorithm, which alternates between estimating the probability of each data point belonging to a Gaussian and updating parameters like mean, variance, and weight to maximize likelihood. Additionally, the covariance structure plays a crucial role in shaping clusters, allowing for spherical, diagonal, tied, or fully adaptable cluster forms, making GMM highly flexible for complex data distributions.


#### The EM Algorithm in GMM
The Expectation-Maximization (EM) algorithm optimizes clustering through an iterative two-step process. In the Expectation Step (E-step), probabilities are assigned to each data point, estimating the likelihood of belonging to a specific Gaussian component. The Maximization Step (M-step) then updates the mean, variance, and weight of each Gaussian to maximize the model’s likelihood. This cycle repeats until convergence, when the parameters stabilize, ensuring an optimal fit for the dataset.


#### Advantages of GMM
Gaussian Mixture Models (GMM) offer probabilistic soft clustering, assigning a probability score to each data point’s cluster membership, which captures uncertainty and overlapping structures. Unlike K-means, GMM allows for flexible cluster shapes, accommodating varying sizes, orientations, and densities. This adaptability makes GMM an excellent choice for clustering complex datasets with overlapping distributions.

#### Basic Code Implementation
Below is a basic Gaussian Mixture Model (GMM) implementation, providing a foundational understanding of how it works in clustering tasks. The code uses GaussianMixture from sklearn.mixture, along with matplotlib for visualization and numpy for numerical operations. It generates 100 random data points in a 2D space, initializes a GMM with three components, and fits the model to the dataset. Cluster assignments are predicted, and the results are visualized with data points color-coded by cluster, while computed cluster centers (means) are highlighted in black. This demonstrates how GMM groups data using probabilistic distributions.

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

![image](https://github.com/user-attachments/assets/ba6c9c55-ed2b-43ef-82a8-543445e64478)

Visualization of clustering results using the Gaussian Mixture Model (GMM). The data points are grouped into three distinct clusters, each represented by a different color. The black points indicate the computed cluster centers (means), highlighting the probabilistic nature of GMM clustering.

### Image Classification
This section applies unsupervised learning for image classification, focusing on distinguishing sea ice from leads using Sentinel-2 imagery. By leveraging clustering algorithms, patterns can be identified and classified without labeled data, improving the analysis and interpretation of remote sensing data efficiently.

#### K-Means Implementation
This Python script applies K-means clustering to a Sentinel-2 satellite image, classifying surface features based on reflectance values from the Red (B4) band. It imports essential libraries, including rasterio for satellite imagery, numpy for numerical operations, KMeans from sklearn.cluster for clustering, and matplotlib.pyplot for visualization. The script loads Band 4 using rasterio.open(), storing pixel values in a NumPy array, while a valid data mask filters out no-data pixels. The image data is reshaped into a 1D array, and K-means is applied with two clusters (n_clusters=2) to distinguish surface types like sea ice and open water. The classified pixel labels are reshaped to the original image dimensions, with masked pixels assigned -1 to maintain data integrity. The clustering results are visualized using plt.imshow(), with distinct colors representing different clusters. This approach enables unsupervised classification of Sentinel-2 imagery, providing insights into surface variations for climate monitoring, environmental research, and land cover classification, without requiring labeled training data.

```python
import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

base_path = "/content/drive/MyDrive/GEOL0069/2425/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for K-means, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

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

del kmeans, labels, band_data, band_stack, valid_data_mask, X, labels_image
```

![image](https://github.com/user-attachments/assets/4dbda879-080b-406b-b1f0-24b97a1d7aa8)

This image shows the K-means clustering result on a Sentinel-2 B4 optical band, grouping pixels into two clusters. Yellow regions likely represent sea ice or land, while darker areas indicate open water or other surface types. The color bar displays cluster labels, with -1 assigned to no-data areas. This classification enhances the distinction of surface features in remote sensing imagery.

#### GMM Implementation
This Python script applies Gaussian Mixture Model (GMM) clustering to a Sentinel-2 B4 optical band using scikit-learn. It reads the B4 band with Rasterio, stacks the data into a NumPy array, and creates a mask to exclude zero-value pixels. The valid data is reshaped and clustered using GMM with two components, allowing for soft clustering, where each pixel is assigned a probability of belonging to each cluster. The results are stored in an array, with -1 for no-data areas, and visualized with Matplotlib, color-coding different clusters. This method is effective for analyzing sea ice, land cover, and environmental patterns in Sentinel-2 imagery.

```python
import rasterio
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Paths to the band images
base_path = "/content/drive/MyDrive/GEOL0069/2425/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for GMM, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

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

![2e7a5d5ebb2325493b5e9bdfa718c6d](https://github.com/user-attachments/assets/d6655404-0987-4593-8fc0-2bcc32270091)

Visualization of Gaussian Mixture Model (GMM) clustering applied to Sentinel-2 Band 4 imagery. The image showcases different clusters identified in the dataset, with the color scale representing distinct cluster labels. This method helps differentiate between various land cover types, such as sea ice, open water, and land surfaces, based on spectral reflectance patterns.


### Altimetry Classification
This section applies unsupervised learning to classify Sentinel-3 altimetry data, focusing on distinguishing sea ice from leads using satellite-derived elevation measurements. This approach enhances the analysis of surface features, improving insights into ice dynamics and oceanographic processes.

#### Read in Functions Needed
This Python script processes Sentinel-3 SAR altimetry data, focusing on waveform characteristics like peakiness and stack standard deviation (SSD) to classify sea ice and leads. It utilizes NumPy, SciPy, Matplotlib, and Scikit-learn for data extraction, cleaning, and clustering using K-Means, DBSCAN, and Gaussian Mixture Models (GMMs). The unpack_gpod function extracts key parameters such as latitude, longitude, waveforms, and backscatter coefficient, interpolating 1Hz data to 20Hz for consistency. calculate_SSD estimates waveform variability using Gaussian curve fitting, aiding classification. The dataset is standardized with StandardScaler to optimize clustering, and NaN values are removed to ensure data integrity. The GMM model, with two components, classifies data into probabilistic clusters, refined iteratively by the Expectation-Maximization (EM) algorithm. Cluster distribution is analyzed using np.unique(), and waveform differences between sea ice and leads are visualized by plotting mean and standard deviation. The blue curve represents sea ice, while the orange curve corresponds to leads, highlighting variations in reflectivity patterns and improving remote sensing classification for environmental and climate studies.

```python
# mean and standard deviation for all echoes
mean_ice = np.mean(waves_cleaned[clusters_gmm==0],axis=0)
std_ice = np.std(waves_cleaned[clusters_gmm==0], axis=0)

plt.plot(mean_ice, label='ice')
plt.fill_between(range(len(mean_ice)), mean_ice - std_ice, mean_ice + std_ice, alpha=0.3)


mean_lead = np.mean(waves_cleaned[clusters_gmm==1],axis=0)
std_lead = np.std(waves_cleaned[clusters_gmm==1], axis=0)

plt.plot(mean_lead, label='lead')
plt.fill_between(range(len(mean_lead)), mean_lead - std_lead, mean_lead + std_lead, alpha=0.3)

plt.title('Plot of mean and standard deviation for each class')
plt.legend()
```

![7c3651827a7ebef8494864dcce54cc8](https://github.com/user-attachments/assets/88e34ef0-0769-4cbb-9a76-073679057a75)


```python
x = np.stack([np.arange(1,waves_cleaned.shape[1]+1)]*waves_cleaned.shape[0])
plt.plot(x,waves_cleaned)  # plot of all the echos
plt.show()
```

![image](https://github.com/user-attachments/assets/f0985c42-840f-49b7-a879-13fbe593d8e2)


```python
# plot echos for the lead cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==1].shape[1]+1)]*waves_cleaned[clusters_gmm==1].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==1])  # plot of all the echos
plt.show()
```

![image](https://github.com/user-attachments/assets/42acd430-03e3-406f-b61f-d0bbdee17288)


```python
# plot echos for the sea ice cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==0].shape[1]+1)]*waves_cleaned[clusters_gmm==0].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==0])  # plot of all the echos
plt.show()
```

![image](https://github.com/user-attachments/assets/21330790-9b6e-4de3-b764-2c4f555f22c6)


### Scatter Plots of Clustered Data
This code visualizes Gaussian Mixture Model (GMM) clustering results on Sentinel-3 altimetry data using scatter plots, where each color represents a different cluster (clusters_gmm). It generates three plots to illustrate relationships between key features: sigma naught (σ₀), Peakiness Parameter (PP), and Stack Standard Deviation (SSD). The first plot shows σ₀ vs. PP, highlighting how backscatter and peakiness are distributed across clusters. The second plot visualizes σ₀ vs. SSD, revealing variations in waveform deviation. The third plot displays PP vs. SSD, helping to distinguish between sea ice and leads. These scatter plots provide a clear representation of cluster separation, enhancing the interpretation of altimetric properties in surface classification.



```python
plt.scatter(data_cleaned[:,0],data_cleaned[:,1],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("PP")
plt.show()
plt.scatter(data_cleaned[:,0],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("SSD")
plt.show()
plt.scatter(data_cleaned[:,1],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("PP")
plt.ylabel("SSD")
```


![image](https://github.com/user-attachments/assets/96752147-485b-4edb-8cd7-4f6cfe140ed3)


![image](https://github.com/user-attachments/assets/cba6ea58-3e8a-44b9-bc7e-34beabcb75f2)


![image](https://github.com/user-attachments/assets/b6875888-69e5-40cb-96e3-b4743761c005)




### Waveform Alignment Using Cross-Correlation
This code aligns waveforms in the GMM cluster (clusters_gmm == 0) using cross-correlation. It first identifies a reference point by locating the peak of the mean waveform for the cluster. Cross-correlation is then applied to determine the shift needed to align each waveform with the first waveform in the cluster. Using np.roll(), the waveforms are adjusted to ensure peak synchronization. Finally, a subset of 10 equally spaced waveforms is plotted, visualizing the alignment and providing insight into waveform consistency within the sea ice cluster.

```python
from scipy.signal import correlate
 
# Find the reference point (e.g., the peak)
reference_point_index = np.argmax(np.mean(waves_cleaned[clusters_gmm==0], axis=0))
 
# Calculate cross-correlation with the reference point
aligned_waves = []
for wave in waves_cleaned[clusters_gmm==0][::len(waves_cleaned[clusters_gmm == 0]) // 10]:
    correlation = correlate(wave, waves_cleaned[clusters_gmm==0][0])
    shift = len(wave) - np.argmax(correlation)
    aligned_wave = np.roll(wave, shift)
    aligned_waves.append(aligned_wave)
 
# Plot aligned waves
for aligned_wave in aligned_waves:
    plt.plot(aligned_wave)
 
plt.title('Plot of 10 equally spaced functions where clusters_gmm = 0 (aligned)')
```

![image](https://github.com/user-attachments/assets/a27f9aef-6c03-40db-80ef-5e370ef71738)


### Compare with ESA data
In the ESA dataset, sea ice = 1 and lead = 2, so 1 is subtracted from all values in flag_cleaned to ensure label compatibility with machine learning models using zero-based indexing. The modified array, flag_cleaned_modified, retains the same structure but with values shifted down by one. To evaluate Gaussian Mixture Model (GMM) clustering, true labels from flag_cleaned_modified are compared with predicted labels from clusters_gmm. A confusion matrix (confusion_matrix(true_labels, predicted_gmm)) summarizes correct and misclassified instances, while a classification report (classification_report(true_labels, predicted_gmm)) provides precision, recall, and F1-score for each class. The results show high accuracy, with 8,856 sea ice and 3,293 lead instances correctly classified, and minimal misclassifications (22 for sea ice, 24 for leads). With an overall accuracy of 100%, GMM effectively distinguishes between the two classes.


```python
Confusion Matrix:
[[8856   22]
 [  24 3293]]

Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      8878
         1.0       0.99      0.99      0.99      3317

    accuracy                           1.00     12195
   macro avg       1.00      1.00      1.00     12195
weighted avg       1.00      1.00      1.00     12195
```
