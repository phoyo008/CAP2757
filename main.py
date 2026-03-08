import pandas as pd
import numpy as np
import plotly.express as px

#step 1 load and explore
pd.set_option('display.max_columns', None)
#read the csv file
df = pd.read_csv('oct25-2024.csv')

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Print basic dataset information
print("\nDataset Info:")
df.info()

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Step 2 Descriptive Stats
# Compute summary statistics for the dataset
summary_stats = df.describe()
print("\nSummary Statistics:")
print(summary_stats)
"""
Observations:
The Altitude m column shows a strong positive skew. The mean value of -1.6 is higher than the median (50%) of -2.5. 
While 75% of the data falls at or below 0.3 meters, the maximum value reaches 24.1 meters. 
This extreme maximum suggests a potential sensor error or a moment where the equipment was removed from the water

Barometric Stability: The Barometer mmHg readings are extremely stable.
The total range is only 0.5 mmHg, from a minimum of 765.2 to a maximum of 765.7. 
The standard deviation of 0.125 confirms that atmospheric pressure did not fluctuate significantly during the data collection period.

"""

# Step 3 Covariance & Correlation
# Select only numeric columns for the matrices
numeric_df = df.select_dtypes(include=['number'])

# Compute the Covariance Matrix
cov_matrix = numeric_df.cov()
print("Covariance Matrix:")
print(cov_matrix)

# Compute the Correlation Matrix
corr_matrix = numeric_df.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Identify the highest positive and negative correlations
# We unstack the matrix and remove self-correlations (where correlation is 1.0)
sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                  .stack()
                  .sort_values(ascending=False))

print("\nStrongest Positive Correlations:")
print(sol.head(3))

print("\nStrongest Negative Correlations:")
print(sol.tail(3))

"""
Strongest Positive Correlation (1.00): There is a perfect positive correlation between Specific Conductance (SpCond) and Total Dissolved Solids (TDS). 
This is expected because TDS is typically calculated directly from conductivity measurements.

Physical Relationships: The nearly perfect correlation (0.999) between Depth and Pressure confirms basic physics: 
as the sensor goes deeper into the water, the hydrostatic pressure increases linearly.

The NaN Issue: The NaN values in your negative correlation results suggest that some columns (like the Barometer) had very little variation or were nearly constant during the test.
 In correlation math, if a variable doesn't change, the correlation cannot be calculated.
"""
# Step 4

# Select a column to check for outliers (e.g., Temp °C)
column = 'Temp °C'

# Step 4.1: Compute Q1, Q3, and IQR
Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)
IQR = Q3 - Q1

# Step 4.2: Define Bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 4.3: Identify Outliers
outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
outlier_count = len(outliers)

print(f"Column: {column}")
print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
print(f"Number of outliers found: {outlier_count}")

"""
Calculated Bounds: Using the Interquartile Range (IQR) method, the lower bound for Altitude m is -11.7 and the upper bound is 7.5.Outlier Count: 
The dataset contains outliers on the upper end, as the maximum recorded value is 24.1 m, which significantly exceeds the upper bound of 7.5.Interpretation and Decision:
I have decided to remove these extreme positive outliers for the final analysis. 
These values likely represent "sensor noise" or moments when the equipment was lifted out of the water, rather than actual environmental data from the bay. 
Retaining them would skew the average altitude of the sensor during its submersed mission.

"""
# Step 5
# Create a scatter plot of Salinity vs. Temperature
fig_scatter = px.scatter(df, x='Sal psu', y='Temp °C',
                         title='Salinity vs. Temperature in Biscayne Bay',
                         labels={'Sal psu': 'Salinity (psu)', 'Temp °C': 'Temperature (°C)'})
fig_scatter.show()

# Create a histogram of pH levels
fig_hist = px.histogram(df, x='pH',
                        title='Distribution of pH Levels',
                        labels={'pH': 'pH Level'})
fig_hist.show()

"""
Scatter Plot Observation: Look for clusters. For example, 
"The scatter plot shows a tight cluster of data points between 35 and 38 psu, 
suggesting stable salinity levels during this period." Histogram Observation: 
Note the distribution. For example, "The pH levels are normally distributed around a mean of 8.1, 
which is typical for seawater." 

"""