import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv("Pollutant_Radar.csv")

# Filter for PM10 readings
pm10_data = df[df['pollutant_id'] == 'PM10'].copy()

# Calculate AQI (simplified calculation based on PM10)
def calculate_aqi(pm10):
    if pm10 <= 50:
        return (pm10 * 50) / 50
    elif pm10 <= 100:
        return 50 + ((pm10 - 50) * 50) / 50
    elif pm10 <= 250:
        return 100 + ((pm10 - 100) * 100) / 150
    elif pm10 <= 350:
        return 200 + ((pm10 - 250) * 100) / 100
    elif pm10 <= 430:
        return 300 + ((pm10 - 350) * 100) / 80
    else:
        return 400 + ((pm10 - 430) * 100) / 80

# Calculate AQI for each PM10 value
pm10_data['AQI'] = pm10_data['pollutant_avg'].apply(calculate_aqi)

# Create the scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=pm10_data, x='pollutant_avg', y='AQI', alpha=0.6)

# Customize the plot
plt.title('PM10 vs Air Quality Index (AQI)', fontsize=14, pad=20)
plt.xlabel('PM10 Concentration (µg/m³)', fontsize=12)
plt.ylabel('Air Quality Index (AQI)', fontsize=12)

# Add a trend line
sns.regplot(data=pm10_data, x='pollutant_avg', y='AQI', 
            scatter=False, color='red', line_kws={'linestyle': '--'})

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()