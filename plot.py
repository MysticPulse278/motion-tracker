import csv
import matplotlib.pyplot as plt
import numpy as np

# Initialize lists to store frame numbers and y positions
frames = []
y_positions = []

# Read the CSV file
with open('trajectories.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        # Append frame number and y position to respective lists
        frames.append(int(row[0]))
        y_positions.append(float(row[2]))

y_max = max(y_positions)
y_positions = y_max - np.array(y_positions)

# Create the plot

plt.plot(frames, y_positions)
plt.xlabel('Frame Number')
plt.ylabel('Y Position')
plt.title('Y Position Over Frame Number')
plt.grid(True)
plt.show()