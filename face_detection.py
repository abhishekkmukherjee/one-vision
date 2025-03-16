import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb

def detect_faces(image):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return faces

def draw_faces(image, faces):
    # Create a copy of the image
    image_with_faces = image.copy()
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image_with_faces, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image_with_faces

def display_image(image, title="Image"):
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    # Path to your image with faces
    image_path = "people_image.jpg"
    
    # Load the image
    original_image, image_rgb = load_image(image_path)
    
    # Display original image
    display_image(image_rgb, "Original Image")
    
    # Detect faces
    faces = detect_faces(original_image)
    
    # Draw rectangles around faces
    image_with_faces = draw_faces(image_rgb, faces)
    
    # Display image with detected faces
    display_image(image_with_faces, "Detected Faces")
    
    print(f"Found {len(faces)} faces!")

if __name__ == "__main__":
    main()