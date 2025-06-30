import tkinter as tk
import pandas as pd
import numpy as np
import random
from PIL import Image, ImageTk

# Load the FER2013 dataset
df = pd.read_csv('fer2013.csv')

# Create the main window
root = tk.Tk()
root.title("Random Convert Images")
root.geometry("800x600")

# Create a centered display area (frame) for images with white background.
# Using pack with expand=True and fill='both' will center the frame.
display_frame = tk.Frame(root, bg="white")
display_frame.pack(expand=True, fill="both", padx=50, pady=50)

# Global list to keep a reference to PhotoImage objects (to prevent garbage collection)
image_labels = []

def random_convert():
    # Clear previous images from the display area
    for widget in display_frame.winfo_children():
        widget.destroy()
    
    global image_labels
    image_labels = []
    
    # Randomly select 4 distinct indices
    indices = random.sample(range(len(df)), 4)
    
    # Create a subframe inside display_frame to center the images horizontally.
    images_frame = tk.Frame(display_frame, bg="white")
    images_frame.pack(expand=True)
    
    # Loop over the 4 random indices and convert the pixel data to images
    for i, idx in enumerate(indices):
        # Get the pixel string and convert it to a NumPy array (48x48)
        pixel_data = df.loc[idx, 'pixels']
        pixel_values = pixel_data.split()
        image_array = np.array(pixel_values, dtype='uint8').reshape(48, 48)
        
        # Create a PIL image from the NumPy array (grayscale mode 'L')
        img = Image.fromarray(image_array, mode='L')
        # Optionally, resize the image (e.g., to 150x150 pixels) for better visibility
        img = img.resize((150, 150), Image.NEAREST)
        
        # Convert the PIL image to a Tkinter PhotoImage
        photo = ImageTk.PhotoImage(img)
        
        # Create a Label widget to hold the image
        label = tk.Label(images_frame, image=photo, bg="white")
        label.photo = photo  # Keep a reference to avoid garbage collection
        # Place each label side by side
        label.pack(side="left", padx=10, pady=10)
        image_labels.append(label)

# Create the "Random convert" button; it is placed at the top of the window.
btn = tk.Button(root, text="Random convert", command=random_convert, font=("Arial", 14))
btn.pack(pady=10)

root.mainloop()
