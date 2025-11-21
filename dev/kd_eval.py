
data = {
    '58': [0.7613, 0.7913, 0.7976, 0.7923, 0.7889],
    '93': [0.8297, 0.8266, 0.8164, 0.8386, 0.8362],
    '126': [0.8436, 0.8546, 0.8165, 0.8344, 0.8342],
    '160': [0.8537, 0.8437, 0.8581, 0.8359, 0.8467],
    '213': [0.8582, 0.8543, 0.8627, 0.8573, 0.8581],
    '327': [0.8699, 0.8696, 0.8588, 0.8655, 0.868]

}
# Compute mean and std for top 3 values in each entry
import numpy as np

for key, values in data.items():
    top3 = sorted(values, reverse=True)[:3]
    mean = np.mean(top3) * 100
    std = np.std(top3) * 100
    print(f"{key}: top3 = {top3}, mean = {mean:.4f}, std = {std:.4f}")
