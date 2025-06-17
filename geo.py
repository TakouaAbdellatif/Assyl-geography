from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.tools import tool

# Using LangChain solution and creact agents
from langgraph.prebuilt import create_react_agent
#from IPython.display import Image, display
import getpass
import os
from langchain_core.pydantic_v1 import BaseModel, Field
# app.py
import streamlit as st
from langchain.chat_models import init_chat_model


#AIzaSyBs_2tjz1v8u-I3voJlLz5J0-tHAaznzQA
if not os.environ.get("GOOGLE_API_KEY"):
  #os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDHnCaa7DGz5pfnt1CG-ZU71UnRL7M2QgM"



llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


class Answer(BaseModel):
    """Answers fields."""

    population: str = Field(description="Donne le nombre d'habitants du pays indiqué.")
    superficie: str = Field(description="Donne la superficie du pays indiqué.")
    histoire: str = Field(description="Donne deux évéènements historiques très marquants de ce pays.")
    cuisine: str = Field(description="Donne les plats les plus connus de ce pays.")
    anecdote: str = Field(description="Donne une anecdote sympatique sur ce pays.")
    

parser = JsonOutputParser(pydantic_object=Answer)
format_instructions = parser.get_format_instructions()

template_with_memory = ChatPromptTemplate.from_messages([
    ("system", "Tu es un géographe et historien très passionné par la géographie et l'histoire des pays."), 
    ("human", "Donne des réponses dans un langage simple."), 
    ("human", (
        "Donne des informations sur le pays indiqué selon les formats d'instructions.\n" +
        "Format instructions {format_instructions}")
    ),
    ('placeholder', '{chat_conversation}')
])

output_parser = RunnableLambda(lambda resp: f"population : {resp['population']}\n superficie: {resp['superficie']} \n histoire: {resp['histoire']} \n cuisine: {resp['cuisine']} \n anecdote: {resp['anecdote']} " )

# ChatBot creation
class Chatbot:
    def __init__(self, llm):
        # This is the same prompt template we used earlier, which a placeholder message for storing conversation history.
        
        chat_conversation_template = template_with_memory.partial(format_instructions=format_instructions)

        # This is the same chain we created above, added to `self` for use by the `chat` method below.
        #self.chat_chain = chat_conversation_template | llm | StrOutputParser()
        self.chat_chain=chat_conversation_template | llm | parser |output_parser

        # Here we instantiate an empty list that will be added to over time.
        self.chat_conversation = []

    # `chat` expects a simple string prompt.
    def chat(self, pays):
        # Append the prompt as a user message to chat conversation.
        self.chat_conversation.append(('human', "Donne les informations sur le pays sauivant:" + pays))
        
        response = self.chat_chain.invoke({'chat_conversation': self.chat_conversation})
        # Append the chain response as an `ai` message to chat conversation.
        self.chat_conversation.append(('ai', response))
        # Return the chain response to the user for viewing.
        return response

    # Clear conversation history.
    def clear(self):
        self.chat_conversation = []





# Title of the app
st.title("Assyl@Geography chatbot")

# User input
user_input = st.text_input("Donnez le nom du pays:", "")
chatbot = Chatbot(llm)
# Placeholder chatbot response logic
def get_response(pays):
    # You can replace this with your Python chatbot agent
    rep=chatbot.chat(pays)
    return rep
def get_history():
    return chatbot.chat_conversation
  
# Display the response
if user_input:
    response = get_response(user_input)
    #history=get_history()
    st.text_area("Pays:", value=response, height=100, max_chars=None)
    
