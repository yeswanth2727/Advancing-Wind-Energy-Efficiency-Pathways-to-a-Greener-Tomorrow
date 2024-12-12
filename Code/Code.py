import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib
import requests
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import  PCA
from sklearn.cluster import  KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error  
from sklearn.model_selection import cross_val_score
import warnings
import py7zr
%matplotlib inline

# Step 1: Download the .7z file from GitHub
url = "https://github.com/yeswanth2727/Advancing-Wind-Energy-Efficiency-Pathways-to-a-Greener-Tomorrow/raw/main/Data%20Set/wind_data.7z"
output_7z_path = "wind_data.7z"

# Download the file
print("Downloading the .7z file...")
response = requests.get(url)
with open(output_7z_path, "wb") as file:
    file.write(response.content)
print("Download complete.")

# Step 2: Extract the .7z file
extracted_path = "extracted_wind_data"
os.makedirs(extracted_path, exist_ok=True)  # Ensure the directory exists

print("Extracting the .7z file...")
with py7zr.SevenZipFile(output_7z_path, mode='r') as archive:
    archive.extractall(path=extracted_path)
print("Extraction complete.")

# Step 3: Locate and read the CSV file
csv_file_path = os.path.join(extracted_path, "wind_data.csv")  # Adjust filename if necessary

if os.path.exists(csv_file_path):
    print("Reading the CSV file...")
    df = pd.read_csv(csv_file_path)
    print("CSV file loaded successfully.")
    print(df.head())  # Display the first few rows of the DataFrame
else:
    print(f"CSV file not found in the extracted directory: {extracted_path}")


# Check the first few rows of the dataframe
df.head()


# Check for duplicate rows
duplicate_rows = df.duplicated()

# Remove duplicate rows
df.drop_duplicates(inplace=True)

df.info()

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['number'])

# Calculate mean, median, standard deviation, skewness, and kurtosis
statistics = numeric_columns.agg(['mean', 'median', 'std', 'skew', 'kurt'])

# Compute the correlation matrix
correlation_matrix = numeric_columns.corr()

# Get basic statistics using describe()
basic_statistics = numeric_columns.describe()

# Print the results
print("Mean, Median, Standard Deviation, Skewness, Kurtosis:")
print(statistics)

print("\nCorrelation Matrix:")
print(correlation_matrix)

print("\nBasic Statistics:")
print(basic_statistics)

# Data Preprocessing
# Convert 'timestamp' to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Drop rows with missing values (if any)
data.dropna(inplace=True)



# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(data['ambient_temperature'], bins=30, kde=True, color='blue')
plt.title('Histogram of Ambient Temperature')
plt.xlabel('Ambient Temperature')
plt.ylabel('Frequency')
plt.savefig('histogram.png')
plt.show()



# Exclude non-numeric columns for correlation matrix
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
correlation_matrix = numeric_data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 6))  # Adjust figure size
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 8})  # Set smaller font size for annotations
plt.title("Correlation Matrix Heatmap", fontsize=14)  # Set smaller title font size
plt.xticks(fontsize=8, rotation=90)  # Rotate x-axis labels by 90 degrees
plt.yticks(fontsize=10, rotation=0)  # Keep y-axis labels horizontal
plt.show()


# Downsample the data for scatter plot
sampled_data = data.sample(n=5000, random_state=42)  # Adjust the n value as needed

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(sampled_data['wind_speed_raw'], sampled_data['active_power_calculated_by_converter'], alpha=0.6, color='green')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Active Power (kW)')
plt.title('Active Power vs. Wind Speed (Sampled Data)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Normalize numerical data for clustering
scaler = StandardScaler()
features_to_scale = [
    'active_power_calculated_by_converter', 'ambient_temperature', 
    'wind_speed_raw', 'generator_speed'
]
data_scaled = scaler.fit_transform(data[features_to_scale])

# -----------------------------------------------
# K-Means Clustering
# -----------------------------------------------

# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

# Plot the elbow method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Plot for K-Means Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.savefig('elbow_plot.png')
plt.show()


# Fit K-Means with optimal clusters (e.g., k=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=data['active_power_calculated_by_converter'], 
    y=data['wind_speed_raw'], 
    hue=data['Cluster'], 
    palette='viridis', 
    alpha=0.7  # Add transparency
)
plt.title('K-Means Clustering Results')
plt.xlabel('Active Power Calculated by Converter')
plt.ylabel('Wind Speed Raw')
plt.legend(title='Cluster', loc='upper right')  # Specify legend location
plt.savefig('cluster_plot.png', dpi=300)  # Save with higher resolution
plt.show()


# -----------------------------------------------
# Linear Regression (Line Fitting)
# -----------------------------------------------

# Using 'wind_speed_raw' to predict 'active_power_calculated_by_converter'
X = data[['wind_speed_raw']].values
y = data['active_power_calculated_by_converter'].values

# Train-test split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Performance Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot Line Fitting
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, label='Actual Data', alpha=0.6)
plt.plot(X_test, y_pred, color='red', label='Predicted Line')
plt.title('Linear Regression: Active Power vs Wind Speed')
plt.xlabel('Wind Speed Raw')
plt.ylabel('Active Power Calculated by Converter')
plt.legend()
plt.savefig('line_fit_plot.png')
plt.show()

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")


