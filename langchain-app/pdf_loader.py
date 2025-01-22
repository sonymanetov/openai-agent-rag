import os

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader


load_dotenv("../.env")

class PdfLoader:
    """
    A class for loading and processing PDF files using PyPDFLoader.
    """

    def __init__(self):
        """
        Initialize the PDF loader. The file path is retrieved from the .env file.
        """
        self.file_path = os.getenv("PDF_PATH")

    def load_and_process_pdf(self):
        """
        Load and process the PDF file. Uses PyPDFLoader to extract documents from the file.

        Returns: 
            A list of documents extracted from the PDF file.
        """
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        return documents
