from .tool import Tool
import os
import subprocess
import shutil  # Import shutil to move files

class Stegosuite(Tool):

    def __init__(self):
        super().__init__("stegosuite")

    def embed_data(self, data, cover_file, stego_file, key="BSC"):
        """
        Embeds data into a cover file to create a stego file.

        :param data: The file path of the data or the message to embed.
        :param cover_file: The file to embed data into.
        :param stego_file: The output file with embedded data.
        :param key: The secret key used for encryption and hiding.
        """
        command = [
            "tools/bin/stegosuite", "embed",
            "-k", key,
            f"-f={data}",
            f"-o={stego_file}",
            cover_file
        ]

        print(f"Running command: {' '.join(command)}")  # Debug: Print the command

        try:
            subprocess.run(command, check=True)
            print("Embedding command executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error: Embedding failed with code {e.returncode}.")
            return

        # Detect the incorrectly placed file
        stego_dir = os.path.dirname(cover_file)  # Get the directory of the cover image
        stego_filename = os.path.basename(cover_file).split('.')[0] + "_embed.jpg"
        auto_stego_file = os.path.join(stego_dir, stego_filename)

        print(f"Checking for auto-generated stego file at: {auto_stego_file}")  # Debug

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


    def extract_data(self, stego_file, output_file, key="BSC"):
        extract_command = ["tools/bin/stegosuite", "extract", stego_file, "-k", key]

        try:
            # Run the extraction command
            subprocess.run(extract_command, check=True)
            print("Extraction command executed successfully.")

            # Move the extracted file to the output location
            move_command = ["mv", "files/stegosuite/attacks/hash.txt", output_file]
            subprocess.run(move_command, check=True)
            print("File moved successfully.")

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
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None