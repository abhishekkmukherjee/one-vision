import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load an image
def load_image(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image '{image_path}'.")
        return None
        
    # Convert from BGR to RGB (OpenCV loads images in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Display the image
def display_image(image, title="Image"):
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show(block=True)  # Use block=True to ensure the plot stays visible

# Apply grayscale conversion
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Apply Gaussian blur
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Apply edge detection
def detect_edges(image, threshold1=100, threshold2=200):
    return cv2.Canny(image, threshold1, threshold2)

# Main function
def main():
    # Path to your image - update this to match your image file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "sample_image.jpg")
    
    print(f"Looking for image at: {image_path}")
    
    # Load the image
    original_image = load_image(image_path)
    
    if original_image is None:
        print("Please make sure your image file exists in the correct location.")
        return
    
    # Display original image
    print("Displaying original image - close the window to continue.")
    display_image(original_image, "Original Image")
    
    # Convert to grayscale
    gray_image = convert_to_grayscale(original_image)
    print("Displaying grayscale image - close the window to continue.")
    display_image(gray_image, "Grayscale Image")
    
    # Apply Gaussian blur
    blurred_image = apply_gaussian_blur(gray_image)
    print("Displaying blurred image - close the window to continue.")
    display_image(blurred_image, "Blurred Image")
    
    # Detect edges
    edges = detect_edges(blurred_image)
    print("Displaying edge detection - close the window to continue.")
    display_image(edges, "Edge Detection")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()