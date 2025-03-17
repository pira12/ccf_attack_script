from .tool import Tool
import os


class Steghide(Tool):

    def __init__(self):
        super().__init__("steghide", "jpg")

    def embed_data(self, data, cover_file, stego_file):
        """
        Embeds data into a cover file to create a stego file.

        :param data: The file path of the data.
        :param cover_file: The file to embed data into.
        :param stego_file: The output file with embedded data.

        flags explained:
        -cf, --coverfile <file>              Cover file
        -mf, --messagefile <file>            Message file
        -sf, --stegofile <file>              Stego file
        """

        # Duplicate the image in png format (jpg not supported)
        # os.system(f"openstego embed -cf {cover_file} -mf {data} -sf {stego_file} -E -p BSC")
        os.system(f'steghide embed -cf {cover_file} -ef {data}  -sf {stego_file} -p ""')

    def extract_data(self, stego_file, output_file):
        """
        Extracts embedded data from a stego file.

        :param stego_file: The file to extract data from.
        :param output_file: The output file to save the extracted data.
        :return: The extracted data.
        """
        # Implementation for extracting data
        os.system(f'steghide extract -sf {stego_file} -p "" -xf {output_file}')

        try:
            with open(f"{output_file}", "r") as f:
                return f.read()
        except FileNotFoundError:
            with open(f"{output_file}", "w") as f:
                f.write("-")
