import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Pollutant_Radar.csv")

# Filter out irrelevant columns
df_filtered = df[["city", "pollutant_id", "pollutant_avg"]]

# Set the plot style
plt.figure(figsize=(15, 8))  # Increased figure size
sns.set_style("whitegrid")

# Create a scatter plot with adjusted parameters
sns.scatterplot(
    data=df_filtered, 
    x="city", 
    y="pollutant_avg", 
    hue="pollutant_id", 
    alpha=0.7,
    s=100  # Increased marker size
)

# Adjust the plot parameters
plt.xlabel("City", fontsize=12)
plt.ylabel("Pollutant Concentration (µg/m³)", fontsize=12)
plt.title("Air Quality Levels Across Cities", fontsize=14, pad=20)

# Rotate and adjust x-axis labels
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()