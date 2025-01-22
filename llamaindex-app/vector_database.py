import logging
import os

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


load_dotenv("../.env")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)

Settings.llm = OpenAI(model=os.getenv("OPENAI_MODEL"))
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")


class VectorDatabase:
    """
    A class for managing vector operations with Qdrant using LlamaIndex.
    """
    def __init__(self):
        self.qdrant_host = os.getenv("QDRANT_HOST")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.vector_size = int(os.getenv("VECTOR_SIZE"))
        self.embeddings = OpenAIEmbedding(
                api_key=self.openai_api_key,
                model="text-embedding-ada-002"
                )

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
                f"Collection '{self.collection_name}' does not exist. Creating a new one..."
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
                    f"Collection '{self.collection_name}' successfully created"
                    )
            except Exception as e:
                logger.error(
                    f"Error creating collection '{self.collection_name}': {e}")
                raise

    def create_index(self, documents):
        """
        Creates an index for the provided documents
        in the specified Qdrant collection.

        Args:
            documents (list): A list of documents to be indexed.

        Returns:
            VectorStoreIndex: An instance of the VectorStoreIndex
            containing the indexed documents.
        """
        try:
            client = self.get_qdrant_client()
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.collection_name
                )

            self.create_collection_if_not_exists(documents)

            logger.info("Uploading documents to the collection...")
            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=512, chunk_overlap=0),
                    TitleExtractor(),
                    self.embeddings,
                ],
                vector_store=vector_store,
            )

            pipeline.run(documents=documents)

            index = VectorStoreIndex.from_vector_store(
                llm=Settings.llm,
                vector_store=vector_store,
                embed_model=Settings.embed_model,
            )

            logger.info(
                f"Index successfully created for collection '{self.collection_name}'."
                )
            return index
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

    def load_index(self, documents):
        """
        Loads the existing index from the specified Qdrant collection.

        Returns:
            VectorStoreIndex: An instance of the VectorStoreIndex
            for the specified collection.
        """
        try:
            client = self.get_qdrant_client()
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.collection_name
                )

            self.create_collection_if_not_exists(documents)

            logger.info(
                f"Index loaded from collection '{self.collection_name}'."
                )

            return VectorStoreIndex.from_vector_store(
                llm=Settings.llm,
                vector_store=vector_store,
                embed_model=Settings.embed_model,
            )
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise
