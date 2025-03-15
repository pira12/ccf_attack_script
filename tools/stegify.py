from .tool import Tool
import os
import subprocess

class Stegify(Tool):

    def __init__(self):
        super().__init__("stegify")

    def embed_data(self, data, cover_file, stego_file):
        """
        Embeds data into a cover file to create a stego file.

        :param data: The file path of the data.
        :param cover_file: The file to embed data into.
        :param stego_file: The output file with embedded data.
        """
        # Run the Stegify encode command
        subprocess.run([
            "tools/bin/stegify_linux_x86-64", "encode",
            "--carrier", cover_file,
            "--data", data,
            "--result", stego_file
        ], check=True)

    def extract_data(self, stego_file, output_file):
        """
        Extracts embedded data from a stego file.

        :param stego_file: The file to extract data from.
        :param output_file: The output file to save the extracted data.
        :return: The extracted data.
        """
        # Run the Stegify decode command
        subprocess.run([
            "tools/bin/stegify_linux_x86-64", "decode",
            "--carrier", stego_file,
            "--result", output_file
        ], check=True)

        try:
            with open(output_file, "rb") as f:
                return f.read()
        except FileNotFoundError:
            with open(output_file, "wb") as f:
                f.write(b"-")
