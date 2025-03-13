import cv2
import numpy as np
import os
import argparse
import subprocess
from PIL import Image, ImageEnhance
from stegano import lsb
from tools.openstego import Openstego


class Attack:
    def __init__(self, image_path, output_folder):
        self.image_path = image_path
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]
        self.output_folder = output_folder
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Image not found!")

    # Function to save image with method name
    def save_image(self, image, method_name):
        filename = os.path.join(
            self.output_folder, f"{method_name}_{self.image_name}.png"
        )
        cv2.imwrite(filename, image)
        print(f"{method_name} applied and saved: {filename}")

    def extract_data(self):
        # Use LSB library to extract the hidden message
        path = os.path.join(self.output_folder, f"embed_data_{self.image_name}.png")
        secret = lsb.reveal(path)
        if secret is None:
            raise ValueError("No hidden message found!")
        msg = secret
        print(f"Extracted message: {msg}")

    # Embed additional data placeholder (doesn't alter image)
    def embed_data(self):
        print("Applying embed_data...")

        secret = lsb.hide(self.image_path, "hello World!")
        secret.save(
            os.path.join(self.output_folder, f"embed_data_{self.image_name}.png")
        )
        print("Data embedded and saved.")

    # Resize Image towards the middle
    def resize(self):
        print("Applying resize...")
        resized = cv2.resize(
            self.image, (self.image.shape[1] // 2, self.image.shape[0] // 2)
        )
        self.save_image(resized, "resized")

    # JPEG Compression
    def compress(self, n=4):
        print("Applying compress...")
        qualities = np.linspace(0, 100, n, dtype=int)
        for quality in qualities:
            filename = os.path.join(
                self.output_folder, f"compressed_{quality}_{self.image_name}.jpg"
            )
            cv2.imwrite(filename, self.image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            print(f"compress applied and saved: {filename} with quality {quality}")

    # Gaussian Blur
    def gaussian_blur(self):
        print("Applying gaussian_blur...")
        blurred = cv2.GaussianBlur(self.image, (9, 9), 0)
        self.save_image(blurred, "gaussian_blur")

    # Noise Addition
    def add_noise(self):
        print("Applying add_noise...")
        noise = np.random.normal(0, 50, self.image.shape).astype(np.int16)
        noisy_image = self.image.astype(np.int16) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        self.save_image(noisy_image, "noise_added")

    # Brightness Adjustment
    def adjust_brightness(self):
        print("Applying adjust_brightness...")
        pil_image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_image)
        brightened = enhancer.enhance(1.5)
        self.save_image(
            cv2.cvtColor(np.array(brightened), cv2.COLOR_RGB2BGR), "brightness_adjusted"
        )

    # Overlay
    def overlay(self):
        print("Applying overlay...")
        overlay_img = np.copy(self.image)
        border_thickness = 50
        margin = 20
        cv2.rectangle(
            overlay_img,
            (border_thickness, border_thickness),
            (
                self.image.shape[1] - border_thickness,
                self.image.shape[0] - border_thickness,
            ),
            (0, 255, 0),
            margin,
        )
        self.save_image(overlay_img, "overlay")

    # Cropping
    def crop(self):
        print("Applying crop...")
        h, w = self.image.shape[:2]
        start_row, start_col = int(h * 0.25), int(w * 0.25)
        end_row, end_col = int(h * 0.75), int(w * 0.75)
        cropped = self.image[start_row:end_row, start_col:end_col]
        self.save_image(cropped, "cropped")

    # Rotating
    def rotate(self):
        print("Applying rotate...")
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 45, 1.0)
        rotated = cv2.warpAffine(self.image, M, (w, h))
        self.save_image(rotated, "rotated")

    # Screenshot (Simulated by resizing and saving)
    def screenshot(self):
        print("Applying screenshot...")
        resized = cv2.resize(self.image, (800, 600))
        self.save_image(resized, "screenshot")

    # Histogram Equalization
    def histogram_equalization(self):
        print("Applying histogram_equalization...")
        img_yuv = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        self.save_image(equalized, "histogram_equalization")


# def generate_stego_image(image_path, secret_data_path, output_folder, tools=None):
#     if tools is None:
#         tools = ["steghide", "stegano"]

#     stego_images = []
#     for tool in tools:
#         stego_image_path = os.path.join(output_folder, f"stego_{tool}.png")
#         if tool == "steghide":
#             command = [
#                 "steghide",
#                 "embed",
#                 "-cf",
#                 image_path,
#                 "-ef",
#                 secret_data_path,
#                 "-sf",
#                 stego_image_path,
#             ]
#         elif tool == "stegano":
#             secret = lsb.hide(image_path, open(secret_data_path).read())
#             secret.save(stego_image_path)
#             command = None
#         else:
#             print(f"Unsupported steganography tool: {tool}")
#             continue

#         if command:
#             try:
#                 subprocess.run(command, check=True)
#                 print(f"Stego image generated successfully using {tool}.")
#             except subprocess.CalledProcessError as e:
#                 print(
#                     f"An error occurred while generating the stego image with {tool}: {e}"
#                 )
#                 continue
#         stego_images.append(stego_image_path)
#     return stego_images


def generate_stego_image(
    secret_data_path="./files/watermarks/hash.txt",
    output_folder="./files",
    image_path="./files/cover_images",
    tools=None,
):
    if tools is None:
        tools = [Openstego()]

    completed = []
    for tool in tools:
        # open files/cover_images and iterate over them
        stego_folder_path = os.path.join(output_folder, tool.name)
        if not os.path.exists(os.path.join(output_folder, tool.name)):
            os.makedirs(stego_folder_path)
            os.makedirs(os.path.join(stego_folder_path, "attacks"))
            os.makedirs(os.path.join(stego_folder_path, "extracted"))

        for cover_image in os.listdir(image_path):
            stego_image_path = os.path.join(stego_folder_path, f"{cover_image}")
            tool.embed_data(secret_data_path, cover_image, stego_image_path)
            print(
                f"Successfully embedded data using {tool.name} in cover images: {completed}",
                end="\r",
            )
        print()


# Main Function
def main(image_path=None, output_folder=None, secret_data_path=None, tools=None):
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    stego_images = generate_stego_image(tools=tools)

    # for stego_image in stego_images:
    #     print(f"\nProcessing attacks for {stego_image}...")
    #     attack = Attack(stego_image, args.output)
    #     attack_methods = [
    #         attack.embed_data,
    #         attack.resize,
    #         attack.compress,
    #         attack.gaussian_blur,
    #         attack.add_noise,
    #         attack.adjust_brightness,
    #         attack.overlay,
    #         attack.crop,
    #         attack.rotate,
    #         attack.screenshot,
    #         attack.histogram_equalization,
    #     ]

    #     for method in attack_methods:
    #         method()

    #     # Verify the embedded data if the --verify flag is set
    #     if args.verify:
    #         try:
    #             attack.extract_data()
    #         except ValueError as e:
    #             print(f"Verification failed for {stego_image}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply multiple transformations to an image."
    )
    parser.add_argument(
        "--image",
        default="./files/cover_images",
        help="Path to the input image (folder)",
    )
    parser.add_argument("--output", default="./files", help="Path to the output folder")
    parser.add_argument(
        "--secret_data",
        default="./files/watermarks",
        help="Path to the secret data to embed",
    )
    parser.add_argument(
        "--tool",
        default="None",
        choices=["steghide", "stegano", "openstego"],
        help="Steganography tool to use",
    )
    parser.add_argument(
        "--verify", help="Verify the embedded data", action="store_true"
    )

    args = parser.parse_args()
    if args.verify:
        attack = Attack(args.image, args.output)
        attack.extract_data()
    else:
        main()
