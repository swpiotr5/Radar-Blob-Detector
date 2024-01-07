import datetime
import numpy as np
import random
import os
import cv2

# Define the size of the board
BOARD_SIZE = 800
IMAGE_HEIGHT = 600
IMAGE_WIDTH = 800

# Define initial attributes for a larger number of objects
objects = [{'shape': 'circle',  # Only generate circles
            'position': (random.randint(0, IMAGE_WIDTH), random.randint(0, IMAGE_HEIGHT)),
            'size': random.randint(2, 5),  # Smaller size
            'color': (255, 255, 255),  # White color
            'direction': (random.uniform(-1, 1), random.uniform(-1, 1))} for _ in range(20)]  # More objects

# ...


# Define the maximum shift per iteration
max_shift = 40  # Increase the maximum shift

# Directory to save the generated images
output_dir = 'pictures'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Start the loop
for i in range(30):
    # Create a blank image of size 800x600
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)  # 3 channels for color

    # Draw an object at each position
    for obj in objects:
        x, y = obj['position']
        color = (255, 255, 255)  # White color
        if obj['shape'] == 'circle':
            cv2.circle(image, (round(x), round(y)), obj['size'], color, -1)

        # Shift the position in the direction of the object
        dx, dy = obj['direction']
        obj['position'] = (x + dx * max_shift, y + dy * max_shift)


    # Generate a filename based on the current time, incremented by i minutes
    filename = (datetime.datetime.now() + datetime.timedelta(minutes=i)).strftime('%Y%m%d%H%M') + '.png'
    output_path = os.path.join(output_dir, filename)

    # Save the image
    cv2.imwrite(output_path, image)