"""
This file is used to create a Chroma database from a CSV file. The database is used to store the reviews and their embeddings.
The embeddings are generated using OpenAI's embedding API. The embeddings are then used to create a Chroma collection. The collection is used to 
retrieve the most similar reviews to a given query.
"""

import dotenv
import os
import sys
import chromadb
import logging
from langchain.document_loaders.csv_loader import CSVLoader
from sentence_transformers import SentenceTransformer
from contextlib import redirect_stdout
import io
import warnings

# Suppress specific FutureWarning from huggingface_hub
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="huggingface_hub.file_download"
)

# Ensure the module path is added to the system path
# This allows us to import modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress Chroma logging if it uses the logging module
# This sets the logging level for the "chromadb" logger to ERROR, suppressing less severe messages
logging.getLogger("chromadb").setLevel(logging.ERROR)


def create_chroma_db():
    """
    This function creates a Chroma database from a CSV file. The database is used to store the reviews and their embeddings.
    The embeddings are generated using OpenAI's embedding API. The embeddings are then used to create a Chroma collection. The collection is used to
    retrieve the most similar reviews to a given query.
    """
    # Load environment variables from a .env file
    dotenv.load_dotenv()

    # Get the path to the CSV file and the Chroma persistence directory from environment variables
    csv_path = os.getenv("REVIEWS_CSV_PATH")
    chroma_path = os.getenv("REVIEWS_CHROMA_PATH")
    print("CSV Path:", csv_path)
    print("Chroma Persist Directory:", chroma_path)

    # Load reviews from the CSV file
    # The CSVLoader loads the CSV file and extracts the "review" column
    # reviews is a list of document objects that contain the text content, metadata, and IDs
    # The IDs are strings and are unique for each document and are created automatically
    reviews = CSVLoader(file_path=csv_path, source_column="review").load()

    """
    Iitialize the SentenceTransformer model for generating embeddings. Embeddings are a list 
    of 1536-dimensional vectors that represent the text content of each document. They are 
    created automatically when the documents are added to the collection. They are used to 
    represent the documents in the vector database. An example of the 1536 dimensions is shown below
    """
    embedding_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    """
    The embedding for this document is 384 dimensions long. The embedding_model.encode(reviews[0].page_content) 
    returns a list of 384 floats. The list is a 384 element long and contains the embedding for the document.
    The length of these embeddings may be different for each document depending on the model used and the 
    document length. The maximum length of the embedding is 1536 and corresponds to a document that is 1000 tokens long.
    """

    # print(embedding_model.encode(reviews[0].page_content))
    # print(len(embedding_model.encode(reviews[0].page_content)))

    # Define a class for the embedding function
    # This class wraps the embedding model's encode method to match the expected signature
    class EmbeddingFunction:
        """
        This class wraps the embedding model's encode method to match the expected signature
        """

        def __call__(self, input):
            return embedding_model.encode(input).tolist()

    # Initialize a Chroma PersistentClient with the specified path
    client = chromadb.PersistentClient(path=chroma_path)

    # Get or create a collection in Chroma with the specified name and embedding function
    collection = client.get_or_create_collection(
        name="reviews_collection", embedding_function=EmbeddingFunction()
    )

    # Extract the text content, metadata, and IDs from the reviews
    texts = [doc.page_content for doc in reviews]
    metadatas = [{"source": doc.metadata["source"]} for doc in reviews]
    ids = [str(i) for i in range(len(texts))]

    # Redirect stdout to suppress unwanted prints during the collection.add call
    f = io.StringIO()
    with redirect_stdout(f):
        # Add the embeddings, metadata, documents, and IDs to the Chroma collection
        collection.add(
            embeddings=EmbeddingFunction()(texts),
            metadatas=metadatas,
            documents=texts,
            ids=ids,
        )

    print("Data added to Chroma collection")
    return collection


# If this script is run directly, call the create_chroma_db function
if __name__ == "__main__":
    create_chroma_db()
