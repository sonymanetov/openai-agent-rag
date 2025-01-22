import logging
import os

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


load_dotenv("../.env")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    A class for managing vector operations with Qdrant.
    """
    def __init__(self):
        self.qdrant_host = os.getenv("QDRANT_HOST")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.vector_size = int(os.getenv("VECTOR_SIZE"))
        self.embeddings = embeddings = OpenAIEmbeddings(
                openai_api_key=self.openai_api_key,
                model="text-embedding-ada-002")

    def get_qdrant_client(self):
        """
        Creates and returns a QdrantClient instance.
        """
        return QdrantClient(url=self.qdrant_host, api_key=self.qdrant_api_key)

    def create_collection_if_not_exists(self, documents):
        """
        Ensures the specified collection exists in Qdrant.
        Creates it if it doesnt exist.
        """
        client = self.get_qdrant_client()

        try:
            client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists.")
        except Exception:
            logger.info(
                f"Collection '{self.collection_name}' does not exist.Creating a new one..."
                )
            try:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )

                self.create_index(documents)

                logger.info(
                    f"Collection '{self.collection_name}' successfully created."
                    )
            except Exception as e:
                logger.error(
                    f"Error creating collection '{self.collection_name}': {e}"
                    )
                raise

    def create_index(self, documents):
        """
        Creates an index for the provided documents
        in the specified Qdrant collection.

        Args:
            documents (list): A list of documents to be indexed.

        Returns:
            Qdrant: An instance of the Qdrant vector store containing
            the indexed documents.
        """
        try:
            client = self.get_qdrant_client()
            self.create_collection_if_not_exists(documents)

            logger.info("Uploading documents to the collection...")
            vectorstore = Qdrant.from_documents(
                documents,
                self.embeddings,
                url=self.qdrant_host,
                api_key=self.qdrant_api_key,
                collection_name=self.collection_name
            )

            logger.info(
                f"Index successfully created for collection '{self.collection_name}'."
                )
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

    def load_index(self, documents):
        """
        Loads the existing index from the specified Qdrant collection.

        Returns:
            Qdrant: An instance of the Qdrant vector store
            for the specified collection.
        """
        try:
            client = self.get_qdrant_client()
            self.create_collection_if_not_exists(documents)

            logger.info(
                f"Index loaded from collection '{self.collection_name}'."
                )
            return Qdrant(
                client=client,
                collection_name=self.collection_name,
                embeddings=self.embeddings)
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise
