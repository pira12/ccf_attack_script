class Tool:

    def __init__(self, name, extension="png"):
        self.name = name
        self.extension = extension

    def embed_data(self, data, cover_file, stego_file) -> None:
        """
        Embeds data into a cover file to create a stego file.

        :param data: The data to embed (can also be file).
        :param cover_file: The file to embed data into.
        :param stego_file: The output file with embedded data.
        """
        # Implementation for embedding data
        pass

    def extract_data(self, stego_file) -> str:
        """
        Extracts embedded data from a stego file.

        :param stego_file: The file to extract data from.
        :return: The extracted data.
        """
        # Implementation for extracting data
        pass
