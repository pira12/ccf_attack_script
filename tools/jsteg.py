from PIL.ImageOps import cover

from .tool import Tool
import os
import subprocess
import shutil  # Import shutil to move files

class Jsteg(Tool):

    def __init__(self):
        super().__init__("jsteg")

    def embed_data(self, data, cover_file, stego_file):
        """
        Embeds data into a cover file to create a stego file.

        :param data: The file path of the data or the message to embed.
        :param cover_file: The file to embed data into.
        :param stego_file: The output file with embedded data.
        """
        command = [
            "tools/bin/jsteg-darwin-arm64", "hide" , cover_file, data, stego_file
        ]

        subprocess.run(command, check=True)



    def extract_data(self, stego_file, output_file):
        """
        Extracts embedded data from a stego file.
        :param stego_file: The file to extract data from.
        :param output_file: The output file to save the extracted data.
        :return: The extracted data.
        """
        command = [
            "tools/bin/jsteg-darwin-arm64", "reveal", stego_file
        ]

        try:
            # Run the extraction command and capture errors
            subprocess.run(command, check=True)
            print("Extraction successful.")

            # Read extracted data
            if os.path.exists(output_file):
                with open(output_file, "rb") as f:
                    return f.read()
            else:
                print("Error: Output file was not created.")
                return None
        except subprocess.CalledProcessError as e:
            print(f"Error: Extraction failed with code {e.returncode}. Wrong key?")
            print(f"Command output: {e.output}")
            return None