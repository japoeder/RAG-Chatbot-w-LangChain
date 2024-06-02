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
* ### Prompt Templates:


  * These are essentially recipes for generating language model prompts.
