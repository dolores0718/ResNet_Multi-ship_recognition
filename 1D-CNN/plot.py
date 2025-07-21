import matplotlib.pyplot as plt

# Data from your training process
epochs = list(range(1, 21))
loss = [1.2804, 1.0683, 0.9518, 0.8456, 0.7577, 0.6754, 0.5941, 0.5345, 0.4805, 0.4326,
        0.3842, 0.3351, 0.3094, 0.2592, 0.2276, 0.2200, 0.1674, 0.1465, 0.1374, 0.1133]
accuracy = [53.68, 57.34, 62.83, 66.82, 69.29, 69.67, 71.22, 72.92, 70.63, 73.30,
            72.49, 72.47, 72.21, 72.37, 72.47, 72.21, 73.14, 72.24, 71.18, 71.09]

# Plot loss in the first figure
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, marker='o', label='Loss', color='#1f77b4')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.grid(True)
plt.legend()

# Plot accuracy in the second figure
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, marker='o', label='Accuracy', color='#aec7e8')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.grid(True)
plt.legend()
plt.show()
