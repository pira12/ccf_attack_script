from PIL.ImageOps import cover

from .tool import Tool
import subprocess

class Jsteg(Tool):

    def __init__(self):
        super().__init__("jsteg", "jpg")

    def embed_data(self, data, cover_file, stego_file):
        """
        Embeds data into a cover file to create a stego file.

        :param data: The file path of the data or the message to embed.
        :param cover_file: The file to embed data into.
        :param stego_file: The output file with embedded data.
        """
        command = [
            "tools/bin/jsteg-linux-amd64", "hide" , cover_file, data, stego_file
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
            "tools/bin/jsteg-linux-amd64", "reveal", stego_file
        ]

        out=open(output_file, 'wb')

        try:
            # Run the extraction command and capture the output
            result = subprocess.run(command, check=True, stdout=out, stderr=out)
        except Exception as e:
            print(e)
        out.close()


