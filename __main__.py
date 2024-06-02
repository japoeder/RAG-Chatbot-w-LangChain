"""
This is the main file for the chatbot.
"""
from langchain_openai import ChatOpenAI
import dotenv

# import os, sys and append current directory so that we can import the chatbot module
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# rom rag_chatbot_portproj.chatbot import *

from rag_chatbot_portproj._templates.visit_review_template import visit_review_template

dotenv.load_dotenv()

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

context = "I had a great stay!"
question = "Did anyone have a positive experience?"

review_obj = visit_review_template()
print(review_obj.format_messages(context=context, question=question))


if __name__ == "__main__":
    print("")
    # chatbot = Chatbot()
    # chatbot.run()
