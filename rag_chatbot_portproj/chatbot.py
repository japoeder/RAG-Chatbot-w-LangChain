"""
Chatbot for RAG project
"""

import dotenv
from langchain_openai import ChatOpenAI
from rag_chatbot_portproj._templates.visit_review_template import visit_review_template

dotenv.load_dotenv()

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

context = "I had a great stay!"
question = "Did anyone have a positive experience?"

review_obj = visit_review_template()
print(review_obj.format_messages(context=context, question=question))
