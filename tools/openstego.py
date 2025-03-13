from .tool import Tool
import os


class Openstego(Tool):

    def __init__(self):
        super().__init__("openstego")

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
        tmp = cover_file[:-4] + "_tmp.png"
        os.system(f"cp {cover_file} {tmp}")
        # Replace unsupported file extension
        stego_file = stego_file.replace(".jpg", ".png")
        os.system(f"openstego embed -cf {tmp} -mf {data} -sf {stego_file} -E")
        os.system(f"rm {tmp}")

    def extract_data(self, stego_file, output_file):
        """
        Extracts embedded data from a stego file.

        :param stego_file: The file to extract data from.
        :param output_file: The output file to save the extracted data.
        :return: The extracted data.
        """
        # Implementation for extracting data
        exit_code = os.system(f"openstego extract -sf {stego_file} -xf {output_file} -p 'Ikwilkaas1'")
        # Read the command output
        # print(f"Exit code: {exit_code}")
        # if exit_code == 0:
        #     with open(f"{output_file}", "w") as f:
        #         f.write("-")
        #     return None

        try:
            with open(f"{output_file}", "r") as f:
                return f.read()
        except FileNotFoundError:
            with open(f"{output_file}", "w") as f:
                f.write("-")
