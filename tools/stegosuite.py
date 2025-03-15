from .tool import Tool
import os
import subprocess
import shutil  # Import shutil to move files

class Stegosuite(Tool):

    def __init__(self):
        super().__init__("stegosuite")

    def embed_data(self, data, cover_file, stego_file, key="BSC", keyfile=None):
        """
        Embeds data into a cover file to create a stego file.

        :param data: The file path of the data or the message to embed.
        :param cover_file: The file to embed data into.
        :param stego_file: The output file with embedded data.
        :param key: The secret key used for encryption and hiding.
        :param keyfile: Path to a file which contains the secret key.
        """
        command = [
            "tools/bin/stegosuite", "embed",
            f"-f={data}",
            cover_file
        ]

        if key:
            command.extend(["-k", key])
        elif keyfile:
            command.extend(["--keyfile", keyfile])

        subprocess.run(command, check=True)

        # Detect the incorrectly placed file
        stego_dir = os.path.dirname(cover_file)  # Get the directory of the cover image
        stego_filename = os.path.basename(cover_file).split('.')[0] + "_embed.jpg"
        auto_stego_file = os.path.join(stego_dir, stego_filename)

        # Move the file to the correct location
        if os.path.exists(auto_stego_file):
            shutil.move(auto_stego_file, stego_file)
            print(f"Stego file moved to: {stego_file}")

            # Delete the original incorrectly placed file
            if os.path.exists(auto_stego_file):  # Double check if the file still exists
                os.remove(auto_stego_file)
                print(f"Deleted incorrect file: {auto_stego_file}")
        else:
            print("Error: Stego file was not created as expected.")


    def extract_data(self, stego_file, output_file, key="BSC", keyfile=None):
        """
        Extracts embedded data from a stego file.

        :param stego_file: The file to extract data from.
        :param output_file: The output file to save the extracted data.
        :param key: The secret key used for encryption and hiding.
        :param keyfile: Path to a file which contains the secret key.
        :return: The extracted data.
        """
        command = ["tools/bin/stegosuite", "extract", stego_file]

        if key:
            command.extend(["-k", key])
        elif keyfile:
            command.extend(["--keyfile", keyfile])

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