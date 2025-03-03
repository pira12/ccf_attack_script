import cv2
import numpy as np
import os
import argparse
from PIL import Image, ImageEnhance

# Function to save image with method name
def save_image(image, method_name, output_folder):
    filename = os.path.join(output_folder, f"{method_name}.png")
    cv2.imwrite(filename, image)
    print(f"{method_name} applied and saved: {filename}")

# Embed additional data placeholder (doesn't alter image)
def embed_data(image, output_folder):
    print("Applying embed_data...")
    save_image(image, "embedded_data", output_folder)

# Resize Image
def resize(image, output_folder):
    print("Applying resize...")
    resized = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    save_image(resized, "resized", output_folder)

# JPEG Compression
def compress(image, output_folder):
    print("Applying compress...")
    filename = os.path.join(output_folder, "compressed.jpg")
    cv2.imwrite(filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    print(f"compress applied and saved: {filename}")

# Gaussian Blur
def gaussian_blur(image, output_folder):
    print("Applying gaussian_blur...")
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    save_image(blurred, "gaussian_blur", output_folder)

# Noise Addition
def add_noise(image, output_folder):
    print("Applying add_noise...")
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    save_image(noisy_image, "noise_added", output_folder)

# Brightness Adjustment
def adjust_brightness(image, output_folder):
    print("Applying adjust_brightness...")
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_image)
    brightened = enhancer.enhance(1.5)
    save_image(cv2.cvtColor(np.array(brightened), cv2.COLOR_RGB2BGR), "brightness_adjusted", output_folder)

# Overlay
def overlay(image, output_folder):
    print("Applying overlay...")
    overlay_img = np.copy(image)
    cv2.rectangle(overlay_img, (50, 50), (200, 200), (0, 255, 0), -1)
    save_image(overlay_img, "overlay", output_folder)

# Cropping
def crop(image, output_folder):
    print("Applying crop...")
    cropped = image[50:300, 50:300]
    save_image(cropped, "cropped", output_folder)

# Rotating
def rotate(image, output_folder):
    print("Applying rotate...")
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    save_image(rotated, "rotated", output_folder)

# Screenshot (Simulated by resizing and saving)
def screenshot(image, output_folder):
    print("Applying screenshot...")
    resized = cv2.resize(image, (800, 600))
    save_image(resized, "screenshot", output_folder)

# Histogram Equalization
def histogram_equalization(image, output_folder):
    print("Applying histogram_equalization...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    save_image(equalized, "histogram_equalization", output_folder)

# Main Function
def main(image_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return

    print("Starting transformations...")
    embed_data(image, output_folder)
    resize(image, output_folder)
    compress(image, output_folder)
    gaussian_blur(image, output_folder)
    add_noise(image, output_folder)
    adjust_brightness(image, output_folder)
    overlay(image, output_folder)
    crop(image, output_folder)
    rotate(image, output_folder)
    screenshot(image, output_folder)
    histogram_equalization(image, output_folder)
    print("All transformations applied successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply multiple transformations to an image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--output", required=True, help="Path to the output folder")
    args = parser.parse_args()
    main(args.image, args.output)
