from .tool import Tool
from stegano import lsb
class Steganotool(Tool):

    def __init__(self):
        super().__init__("Stegano", "png")

    def embed_data(self, data, cover_file, stego_file):
        """
        Embeds data into a cover file to create a stego file.

        :param data: The file path of the data or the message to embed.
        :param cover_file: The file to embed data into.
        :param stego_file: The output file with embedded data.
        """

        secret = lsb.hide(cover_file, "993997e01d6e3a4d2ac39b8a0e4e09296e36b88f1eced46fe887b35906a0d00a")
        print(f"✅the data '993997e01d6e3a4d2ac39b8a0e4e09296e36b88f1eced46fe887b35906a0d00a'")
        secret.save(stego_file)
        print(f"✅ Message hidden successfully in {stego_file}")


    def extract_data(self, stego_file, output_file):
        """
        Extracts embedded data from a stego file.
        :param stego_file: The file to extract data from.
        :param output_file: The output file to save the extracted data.
        :return: The extracted data.
        """
        try:
            # Attempt to reveal the hidden message
            hidden_message = lsb.reveal(stego_file)

            if hidden_message is not None:
                # Write the hidden message to the output file
                with open(output_file, "w") as f:
                    f.write(hidden_message)
                print(f"🔍 Hidden message extracted and saved to {output_file}")
                return hidden_message
            else:
                with open(output_file, "w") as f:
                    f.write("-")
                print("⚠️ No hidden message found!,'-' was written to output file")
                return None

        except Exception as e:
            print(f"An error occurred: {e}")
            with open(output_file, "w") as f:
                f.write("-")
            print("⚠️ No hidden message found!,'-' was written to output file")
            return None