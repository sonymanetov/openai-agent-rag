import logging
import os

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata

from pdf_loader import PdfLoader
from vector_database import VectorDatabase


load_dotenv("../.env")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def is_cat_obese(weight_kg: float) -> str:
    """
    Determines the weight status of a cat (normal, underweight, or obese).
    
    :param weight_kg: Weight of the cat in kilograms.
    :return: Message describing the cat's weight status.
    """
    try:
        weight_kg = float(weight_kg)
    except ValueError:
        return "Error: Weight must be a valid number (e.g., 4.5)."

    if weight_kg < 3.5:
        return f"A cat weighing {weight_kg} kg is underweight. Consult a veterinarian."
    elif 3.5 <= weight_kg <= 5.5:
        return f"A cat weighing {weight_kg} kg is within the normal range."
    else:
        return f"A cat weighing {weight_kg} kg is obese. Consult a veterinarian."


def get_tools(index) -> list:
    """
    Creates a list of tools for the agent.
    
    :param index: LlamaIndex instance for document retrieval.
    :return: List of tools.
    """

    query_engine_tools = [
        QueryEngineTool(
            query_engine=index,
            metadata=ToolMetadata(
                name="Document",
                description="Useful for questions related to specific facts",
            ),
        ),

        FunctionTool.from_defaults(
        fn=is_cat_obese, name="Weight"
        )
    ]

    return query_engine_tools


def get_agent_instructions() -> str:
    """
    Provides instructions for the agent to always respond in Russian.
    
    :return: Instruction string.
    """
    return (
        "Ты умный и добрый ассистент, который ВСЕГДА отвечает на русском языке. "
        "Если пользователь задаёт вопрос, дай ответ на русском."
        "Верни пользователю ответы на все заданные им вопросы."
        "Если пользователь предоставляет в вопросе вес кошки, передавай на обработку ТОЛЬКО число."
        "Пожалуйста, отвечай пользователю подробно и доброжелательно."
    )


def main():
    try:
        logger.info("Initializing Qdrant...")

        vector_db = VectorDatabase()
        pdf_loader = PdfLoader()

        pdf_path = os.getenv("PDF_PATH")
        model_name = os.getenv("OPENAI_MODEL")

        try:
            documents = pdf_loader.load_and_process()
            index = vector_db.load_index(documents)
        except Exception as e:
            raise Exception(
                f"Error occurred while loading index: {e}"
            )

        llm = OpenAI(model=model_name, api_key=os.getenv("OPENAI_API_KEY"))

        logger.info("Chatbot is ready! Enter a query (or type 'e' to quit):")
        while True:
            query = input("> ")
            if query.lower() == "e":
                break

            try:
                query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

                tools = get_tools(query_engine)
                instructions = get_agent_instructions()

                agent = ReActAgent.from_tools(
                    tools,
                    llm=llm,
                    verbose=True,
                    system_prompt=instructions)

                response = agent.chat(query)

                logger.info(f"Response: {response.response}")

            except Exception as e:
                logger.error(f"Error during query execution: {e}")

    except Exception as e:
        logger.critical(f"Critical error: {e}")


if __name__ == "__main__":
    main()
