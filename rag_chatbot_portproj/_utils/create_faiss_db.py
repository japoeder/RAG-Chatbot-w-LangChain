"""
This file is used to create a FAISS database from a CSV file. The database is used to store the reviews and their embeddings.
The embeddings are generated using OpenAI's embedding API. The embeddings are then used to create a FAISS index. The index is used to 
retrieve the most similar reviews to a given query.
"""

import dotenv
import os
import sys
import faiss
import numpy as np
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_faiss_db():
    """
    This function creates a FAISS database from a CSV file. The database is used to store the reviews and their embeddings.
    The embeddings are generated using OpenAI's embedding API. The embeddings are then used to create a FAISS index. The index is used to
    retrieve the most similar reviews to a given query.
    """
    dotenv.load_dotenv()

    # Debug: Print environment variables to check paths
    csv_path = os.getenv("REVIEWS_CSV_PATH")
    print("CSV Path:", csv_path)

    loader = CSVLoader(file_path=csv_path, source_column="review")
    reviews = loader.load()

    # Debug: Print first few reviews to check data loading
    # print("Sample Reviews Loaded:", reviews[:5])

    # Generate embeddings
    embeddings = OpenAIEmbeddings()
    vectors = np.array(
        embeddings.embed_documents([doc.page_content for doc in reviews])
    )

    # Create FAISS index
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    return index, reviews
