import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv("Pollutant_Radar.csv")

# Filter for PM2.5 readings
pm25_data = df[df['pollutant_id'] == 'PM2.5'].copy()

# Calculate AQI (based on PM2.5)
def calculate_aqi_pm25(pm25):
    if pm25 <= 30:
        return (pm25 * 50) / 30
    elif pm25 <= 60:
        return 50 + ((pm25 - 30) * 50) / 30
    elif pm25 <= 90:
        return 100 + ((pm25 - 60) * 100) / 30
    elif pm25 <= 120:
        return 200 + ((pm25 - 90) * 100) / 30
    elif pm25 <= 250:
        return 300 + ((pm25 - 120) * 100) / 130
    else:
        return 400 + ((pm25 - 250) * 100) / 130

# Calculate AQI for each PM2.5 value
pm25_data['AQI'] = pm25_data['pollutant_avg'].apply(calculate_aqi_pm25)

# Create the scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=pm25_data, x='pollutant_avg', y='AQI', alpha=0.6)

# Customize the plot
plt.title('PM2.5 vs Air Quality Index (AQI)', fontsize=14, pad=20)
plt.xlabel('PM2.5 Concentration (µg/m³)', fontsize=12)
plt.ylabel('Air Quality Index (AQI)', fontsize=12)

# Add a trend line
sns.regplot(data=pm25_data, x='pollutant_avg', y='AQI', 
            scatter=False, color='red', line_kws={'linestyle': '--'})

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Add a small delay before showing the plot
plt.pause(0.1)  # Add this line before plt.show()

# Show the plot
plt.show()