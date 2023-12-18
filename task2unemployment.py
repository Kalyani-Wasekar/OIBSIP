# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Replace 'YOUR_DATA_URL' with the actual URL of your CSV file
data_url = "https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india/download?datasetVersionNumber=5"
unemployment_data = pd.read_csv(data_url)

# Display the first few rows of the dataset
print(unemployment_data.head())

# Data Cleaning and Preprocessing (if needed)
# You can add additional steps here based on your data

# Data Visualization
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='UnemploymentRate', data=unemployment_data)
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.show()

# Time Series Analysis
result = seasonal_decompose(unemployment_data['UnemploymentRate'], model='additive', period=12)
result.plot()
plt.show()
