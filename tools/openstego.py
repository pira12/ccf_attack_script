from .tool import Tool
import os


class Openstego(Tool):

    def embed_data(self, data, cover_file, stego_file):
        """
        Embeds data into a cover file to create a stego file.

        :param data: The data to embed (can also be file).
        :param cover_file: The file to embed data into.
        :param stego_file: The output file with embedded data.
        """
        # Implementation for embedding data
        os.system(f"openstego embed -mf {cover_file} -cf {data} -sf {stego_file}")

    def extract_data(self, stego_file):
        """
        Extracts embedded data from a stego file.

        :param stego_file: The file to extract data from.
        :return: The extracted data.
        """
        # Implementation for extracting data
        pass
