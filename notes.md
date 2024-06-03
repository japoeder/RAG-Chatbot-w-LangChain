### LangChain

* LLM is the core componen
* Modular interface
* Can chat directly with LLM objects, but more common to use a chat model
* Chat models can interface with chat messages as opposed to raw text

  * Raw text is passed basically using the invoke() method
  * It is possible to provide instructions for the LLM in the raw text data.
* ### Chat messages:


  * Additional detail provided about the kind of messages
  * 'role' and 'content' properties
  * Tells the LLM who is sending and what the message itself is
  * Common message types:
    * HumanMessage: message from the user interacting with the model
    * AIMessage: message from a language model
    * SystemMessage: how the model should behave
    * FunctionMessage: will learn about later
    * ToolMessage: learn about later
* ### Methods:


  * .stream(): returns the response one token at a time
  * .batch(): accepts a list of messages that the LLM responds to in one call
  * .ainvoke(): asynchronous call of invoke().  All methods have this option.
  * ***any method a chat model supports can be used when calling chains.***
* ### Prompt Templates:


  * These are essentially recipes for generating language model prompts.
  * ***3 CORE COMPONENTS:***
    * ***Instruction set: This is instruction for the bot on how it should behave and what's required of the bot***
      * This is essentially telling the bot for this project that it's goal is to answer questions about customer reviews.
    * ***Context: This is the detail relevant to a given question.***
      * For this project it's the reviews that patients have left.
    * ***Question / message: This is the actual thing we want the chatbot to provide a response for.***
      * For this project it's asking if patients were generally happy with their stay.
* ### Chains


  * This is the glue that connects LangChain objects.
  * The recommended way to build chains is with LCEL:
    * LangChain Expression Language
  * Arbitrary in length and use the pipe ('|') to create links
    * You're basically setting up a pipeline
* ### Output processing


  * StrOutputParser() - use this method to extract the response text
* ### Retrieval objects


  * Trying to fit all detail into the ***context*** window may be impractical (e.g. 1M individual reviews)
  * If you could fit them all in there, it's not guaranteed that the correct reviews will be used when answering questions.
  * Manually managing context also doesn't scale well, so this is where ***retrievers*** come in.
  * 








asdf
