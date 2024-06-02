"""
Template for the visit review chatbot
"""

from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)


def visit_review_template():
    """
    Returns a prompt template for the visit review chatbot
    """

    review_system_template_str = """Your job is to use patient
    reviews to answer questions about their experience at a
    hospital. Use the following context to answer questions.
    Be as detailed as possible, but don't make up any information
    that's not from the context. If you don't know an answer, say
    you don't know.

    {context}
    """

    review_system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"], template=review_system_template_str
        )
    )

    review_human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=["question"], template="{question}")
    )

    messages = [review_system_prompt, review_human_prompt]

    return ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=messages,
    )
