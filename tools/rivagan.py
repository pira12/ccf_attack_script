from .tool import Tool
import os
import subprocess

class Rivagan(Tool):
    def __init__(self):
        super().__init__("rivagan")

    def embed_data(self, data, cover_file, stego_file, method="dwtDct", watermark_type="bytes"):
        """
        Embeds data into a cover file to create a stego file.

        :param data: The file path of the data.
        :param cover_file: The file to embed data into.
        :param stego_file: The output file with embedded data.
        """
        subprocess.run([
            "tools/bin/invisible-watermark/invisible-watermark", "-v",
            "-a", "encode",
            "-t", watermark_type,
            "-w", data,
            "-m", method,
            "-o", stego_file,
            cover_file,
        ], check=True)

    def extract_data(self, stego_file, output_file, method="dwtDct", watermark_type="bytes", length="216"):
        """
        Extracts an invisible watermark from an image or video file.

        :param watermarked_file: The file containing the invisible watermark.
        :param output_file: The output file to save the extracted watermark.
        :return: The extracted watermark data.
        """
        try:
            subprocess.run([
                "tools/bin/invisible-watermark/invisible-watermark", "-v",
                "-a" , "decode",
                "-t", watermark_type,
                "-m", method,
                "-l", length,
                "-o", output_file,
                stego_file
            ], check=True)

            with open(output_file, "rb") as f:
                return f.read()
        except subprocess.CalledProcessError as e:
            print(f"Error during extraction execution: {e}")
            return None
        except FileNotFoundError:
            print(f"Output file not found: {output_file}")
            with open(output_file, "wb") as f:
                f.write(b"-")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
