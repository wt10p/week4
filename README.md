# GEOL0069_Satellite Echo Classification
This repository uses machine learning to classify echoes into **sea ice** and **leads**. It calculates average echo shapes, computes standard deviations, and evaluates results against ESA's classification using a confusion matrix.


This project combines Sentinel-3 (OLCI & SRAL) and Sentinel-2 optical data, using unsupervised learning to classify **sea ice** and **leads**. It aims to create an automated pipeline for enhanced Earth Observation (EO) analysis by integrating satellite datasets and applying machine learning to classify environmental features.

Goals：
Learn to colocate multi-sensor satellite data.

Apply unsupervised learning to classify environmental features like sea ice and leads.

Combine altimetry and optical data for enhanced classification.

Evaluate results using confusion matrices and accuracy metrics.

Satellite data analysis is a dynamic field, and methods may vary by dataset. This project offers a practical guide to integrating remote sensing and machine learning.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

This project utilizes key Python libraries and geospatial tools to process, analyze, and classify Earth Observation (EO) data. Below are the major dependencies used:

* NumPy – Numerical computations and matrix operations
* Pandas – Data manipulation and tabular processing
* Matplotlib – Visualization of classification results
* Rasterio – Handling Sentinel-2 geospatial raster data
* netCDF4 – Processing Sentinel-3 altimetry data
* Scikit-Learn – Machine learning models (K-Means, GMM)
* Folium – Geospatial data visualization
* Shapely – Geometric operations for colocation analysis
* Requests – API calls for Sentinel-3 metadata retrieval

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data -->
## Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data
This section focuses on co-locating Sentinel-2 and Sentinel-3 data, combining Sentinel-2’s high spatial resolution with Sentinel-3’s broader coverage and altimetry insights to improve Earth observation analysis. This integration enhances environmental monitoring, particularly for sea ice and lead classification. The next steps outline how to identify, align, and analyze these datasets effectively.

### Step 0: Read in Functions Needed

This process begins by loading essential functions to efficiently retrieve Sentinel-2 and Sentinel-3 metadata, following the Week 3 approach. Google Drive is mounted in Google Colab for seamless file access. Using requests, pandas, shapely, and folium, the script fetches, processes, and visualizes data from the Copernicus Data Space Ecosystem. Authentication is handled via access tokens, and data is queried by date range, location, and cloud cover. Sentinel-3 OLCI, SRAL, and Sentinel-2 optical data are retrieved, with products downloadable by unique IDs. Geospatial footprints are processed to match images based on geographic overlap, and results are visualized using interactive maps. Time-handling functions ensure accurate timestamp formatting, enabling smooth integration with scientific research and Earth observation projects. 🚀

<!-- Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case) -->
### Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case)
This process co-locates Sentinel-2 and Sentinel-3 OLCI data by retrieving their metadata separately, following the Week 3 approach. The goal is to identify common observation locations, creating sentinel3_olci_data and sentinel2_data for further analysis. Authentication is required to obtain and refresh access tokens before setting a date range and file path for retrieval. The script queries Sentinel-3 OLCI and Sentinel-2 optical data using query_sentinel3_olci_arctic_data() and query_sentinel2_arctic_data(), applying a 0–10% cloud cover filter for Sentinel-2. Metadata is saved as sentinel3_olci_metadata.csv and sentinel2_metadata.csv for alignment and analysis. To enhance visualization, both datasets are displayed in structured tables using IPython's display(), making it easier to inspect key details like product IDs, acquisition times, geospatial footprints, and cloud cover percentages.

![422d13d80a9193db1c3d56e377ac803](https://github.com/user-attachments/assets/cfcaa103-e028-45e3-bcf9-b91b6957116c)

The table displays the metadata retrieved for Sentinel-3 OLCI images within the specified time range. It includes essential attributes such as unique product IDs, names, content types, origin dates, modification dates, and storage paths. This metadata is crucial for identifying and accessing relevant satellite data for further analysis and co-location with Sentinel-2.

![7a75d1a91255c7e1a9145b45bb2fb72](https://github.com/user-attachments/assets/b428d280-5b6a-4ce4-a588-f9d46936ab1d)
This table displays metadata retrieved for Sentinel-2 images using the Copernicus Data Space API. It includes details such as product IDs, content type, content length, acquisition dates, publication and modification timestamps, online availability, and storage paths. This dataset is essential for analyzing and identifying relevant Sentinel-2 imagery based on specific timeframes and geospatial locations.

#### Co-locate the data
This process identifies co-location pairs by matching Sentinel-2 and Sentinel-3 OLCI data based on their geo_footprint. Metadata timestamps from ContentDate are standardized using eval(), pd.to_datetime(), and make_timezone_naive() for consistent time comparisons. The check_collocation() function detects overlapping observations within a 10-minute window, aligning both datasets for geospatial analysis. The resulting results DataFrame contains matched records where both satellites observed the same location. To visualize co-located data, plot_results() maps the first five observations using folium, and IPython's display() renders an interactive map, allowing users to inspect overlapping locations.

![image](https://github.com/user-attachments/assets/f136ac9d-181f-42e3-9079-af0e87e2fdbf)
The table displays the first five rows of the collocated dataset, showing matched Sentinel-2 and Sentinel-3 OLCI observations. Each row contains details about the two satellites, including their unique IDs, footprints (geographical coverage), and the time range during which their observations overlap within a 10-minute window. This output helps verify the successful identification of collocated satellite data for further analysis.

![image](https://github.com/user-attachments/assets/e0bb2933-f303-4a27-ace3-e0dab44e97eb)
This interactive map visualization displays the geographical footprints of the first five collocated satellite observations from Sentinel-2 and Sentinel-3 OLCI. The overlapping satellite data areas are highlighted, showing the regions where both satellites have captured observations within the specified time window.


<!-- Proceeding with Sentinel-3 OLCI Download -->
#### Proceeding with Sentinel-3 OLCI Download
Next, the focus shifts to retrieving Sentinel-3 OLCI data, following the same structured approach used for Sentinel-2 to ensure consistency. By applying the same filename conversion logic, the required datasets are systematically accessed and downloaded from the Copernicus Dataspace, ensuring seamless integration into the analysis pipeline. This step facilitates the download of a specific Sentinel-3 OLCI product. The download_dir variable defines the target directory, while product_id and file_name are extracted from the results DataFrame, selecting the first product for download. The download_single_product() function, along with an access_token, ensures secure retrieval of the satellite data, storing it in the designated directory for further analysis. Users can modify product_id, file_name, and download_dir to customize their downloads.

#### Sentinel-3 SRAL
This process extends co-location analysis by integrating Sentinel-3 SRAL altimetry data alongside Sentinel-2 and Sentinel-3 OLCI observations. The query_sentinel3_sral_arctic_data() function retrieves SRAL metadata for a specified date range using an access token, storing it in s3_sral_metadata.csv for further processing. Previously saved metadata files (s3_sral_metadata.csv and sentinel2_metadata.csv) are loaded with pd.read_csv(), and ContentDate timestamps are standardized using eval(), pd.to_datetime(), and make_timezone_naive() for consistency. The check_collocation() function identifies overlapping Sentinel-2 and Sentinel-3 SRAL observations within a 10-minute window, storing results in a results DataFrame. To visualize co-located data, plot_results() maps the top five matches using GeoJSON footprints, and IPython's display() renders an interactive world map, allowing users to analyze spatial relationships and assess co-location accuracy.

![image](https://github.com/user-attachments/assets/908fe20f-02df-403e-9937-32f8b527bc1b)
This interactive map visualizes the collocation of Sentinel-2 and Sentinel-3 SRAL satellite data. The blue outlines represent the geographical footprints of the detected overlaps, illustrating how the two satellite datasets align over the Arctic region. This visualization helps assess spatial intersections and validate the effectiveness of the collocation process.


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
