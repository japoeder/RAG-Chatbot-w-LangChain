"""
This is the main file for the chatbot.
"""
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import dotenv
import os
import sys

# import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from rag_chatbot_portproj._templates.visit_review_template import (
#     visit_review_template_base,
#     visit_review_template_chain,
# )
from rag_chatbot_portproj._utils.create_faiss_db import create_faiss_db
from rag_chatbot_portproj._utils.create_chroma_db import create_chroma_db

dotenv.load_dotenv()

###############################

chat_model = ChatOpenAI(model="gpt-4", temperature=0)

# context = "I had a great stay!"
# question = "Did anyone have a positive experience?"

# review_obj = visit_review_template_base()
# print(review_obj.format_messages(context=context, question=question))
# print("")
# print("############################################################")
# print("")
# review_obj = visit_review_template_chain()
# print(review_obj.invoke({"context": context, "question": question}))

# Create the FAISS vector database
faiss_index, faiss_reviews = create_faiss_db(["rag_chatbot_portproj/_data/reviews.csv"])

# Create the Chroma vector database
chroma_collection = create_chroma_db()

print("")
print("############################################################")
print("")

question = """Has anyone complained about communication with the hospital staff?"""
print(question)
print("")

# # Perform similarity search with FAISS
# embeddings = OpenAIEmbeddings()
# query_embedding = np.array(embeddings.embed_query(question)).reshape(1, -1)
# D, I = faiss_index.search(query_embedding, k=3)

# print("FAISS Results:")
# # Print the distinct results from FAISS
# for i, idx in enumerate(I[0]):
#     print(f"Result {i+1}:")
#     print(faiss_reviews[idx].page_content)
#     print("")

print("############################################################")
print("")

# Perform similarity search with Chroma
embedding_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
query_vector = embedding_model.encode(question).tolist()
res = chroma_collection.query(
    query_embeddings=[query_vector],
    n_results=3,
    include=["distances", "embeddings", "documents", "metadatas"],
)

print("Chroma Results:")
# Print the distinct results from Chroma
for i, doc in enumerate(res["documents"][0]):
    print(f"Result {i+1}:")
    print(doc)
    print("")

if __name__ == "__main__":
    print("")
    # chatbot = Chatbot()
    # chatbot.run()
