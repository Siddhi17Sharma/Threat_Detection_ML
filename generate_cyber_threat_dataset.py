import pandas as pd
import numpy as np

# Parameters
num_samples = 1000
np.random.seed(42)

# Generate synthetic features (e.g., network traffic data)
feature_1 = np.random.uniform(0, 100, num_samples)
feature_2 = np.random.uniform(0, 500, num_samples)
feature_3 = np.random.uniform(0, 1000, num_samples)
feature_4 = np.random.uniform(0, 10, num_samples)

# Simulate labels (1 for threat, 0 for no threat) with some noise
labels = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])

# Create a DataFrame
df = pd.DataFrame({
    'feature_1': feature_1,
    'feature_2': feature_2,
    'feature_3': feature_3,
    'feature_4': feature_4,
    'label': labels
})

# Save to CSV
df.to_csv('cyber_threat_dataset.csv', index=False)
print('Synthetic cyber threat dataset saved as cyber_threat_dataset.csv')
