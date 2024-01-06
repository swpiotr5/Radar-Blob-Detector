import cv2
import datetime
import numpy as np
import random
import os

# Define the size of the board
BOARD_SIZE = 800

# Define initial attributes for 10 objects
objects = [{'shape': random.choice(['circle', 'rectangle', 'ellipse']),
            'position': (random.randint(0, BOARD_SIZE), random.randint(0, BOARD_SIZE)),
            'size': random.randint(10, 50),
            'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))} for _ in range(10)]

# Define the maximum shift per iteration
max_shift = 40  # Increase the maximum shift

# Directory to save the generated images
output_dir = 'pictures'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Start the loop
for i in range(30):
    # Create a blank image of size 800x600
    image = np.zeros((600, 400, 3), dtype=np.uint8)  # 3 channels for color

    # Draw an object at each position
    for obj in objects:
        x, y = obj['position']
        color = (255, 255, 255)  # White color
        if obj['shape'] == 'circle':
            cv2.circle(image, (x, y), obj['size'], color, -1)
        elif obj['shape'] == 'rectangle':
            cv2.rectangle(image, (x, y), (x + obj['size'], y + obj['size']), color, -1)
        elif obj['shape'] == 'ellipse':
            cv2.ellipse(image, (x, y), (obj['size'], obj['size']), 0, 0, 360, color, -1)

        # Shift the position randomly
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        obj['position'] = (x + dx, y + dy)

    # Generate a filename based on the current time, incremented by i minutes
    filename = (datetime.datetime.now() + datetime.timedelta(minutes=i)).strftime('%Y%m%d%H%M') + '.png'
    output_path = os.path.join(output_dir, filename)

    # Save the image
    cv2.imwrite(output_path, image)