import os

from dotenv import load_dotenv
from llama_index.core import Document, SimpleDirectoryReader


load_dotenv("../.env")

class PdfLoader:
    """
    A class for loading and processing PDF files using SimpleDirectoryReader.
    """

    def __init__(self):
        self.file_path = os.getenv("PDF_FOLDER")

    def load_and_process(self):
        """
        Load and process the PDF file. Extracts text from pages and returns a list of Document objects.

        Returns:
            A list of Document objects containing the text of PDF pages.
        """
        try:
            reader = SimpleDirectoryReader(input_dir=self.file_path)
            documents = reader.load_data()
            return documents
        except Exception as e:
            raise RuntimeError(f"Failed to load and process PDF: {e}")
