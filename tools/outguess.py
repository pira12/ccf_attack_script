from PIL.ImageOps import cover
from .tool import Tool
import os

from .tool import Tool
import os


class Outguess(Tool):

    def __init__(self):
        super().__init__("outguess", "jpg")

    def embed_data(self, data, cover_file, stego_file):
        """
        Embeds data into a cover file to create a stego file.

        :param data: The file path of the data.
        :param cover_file: The file to embed data into.
        :param stego_file: The output file with embedded data.

        flags explained:
        -d, specifies payload/message <file>   Message file
        -t, indicates cover image <file>      Cover image file
        """

        os.system(f"outguess -d {data} -t {cover_file} {stego_file}")

    def extract_data(self, stego_file, output_file):

        """
        Extracts embedded data from a stego file.
        flags explained:
            -r, retrieve    Stego file

        :param stego_file: The file to extract data from.
        :param output_file: The output file to save the extracted data.
        :return: The extracted data.
        """
        # Implementation for extracting data
        exit_code = os.system(f"outguess -r {stego_file} {output_file}")

        try:
            with open(f"{output_file}", "rb") as f:
                return f.read()
        except FileNotFoundError:
            with open(f"{output_file}", "w") as f:
                f.write("-")

