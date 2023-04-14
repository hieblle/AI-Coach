import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
import pandas as pd
import numpy as np


# streamlit settings/page configuration
st.set_page_config(page_title="GPT-3 Chatbot", page_icon=":rocket:", layout="wide")
#st.markdown("# Main page")
st.sidebar.markdown("#  Main page")
st.title("AI-Coach")

st.sidebar.header("Instuctions")
st.sidebar.info("This is a demo of an AI-Coach. You can ask it anything, but it's best at answering questions about the world and its current state.")
API_0 = "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"
modus = st.sidebar.selectbox(label= 'Modus', options=["Journaling questions", "Analysis", "Therapy Chatbot"])
# automatisches auswählen? Word embeddings immer ada.
model = st.sidebar.selectbox(label= 'Model', options=["gpt-3.5-turbo", "text-davinci-003", "text-ada-001"])
st.sidebar.info("Select a model according your task.")

# get text input from user

def get_text():

    journal_text = st.text_input("Please import your Journal entries here as plain text: ", placeholder="Your journal text here.") #, label_visibility="hidden")
    return journal_text


def journal_analytics(journal_entry):
    prompt = f"""You are in the position of a therapist reading over journal entries.
                Goals:
                1. Help people to overcome the things which are holding them back.
                2. Discover recurring patterns.
                3. Read over journal entries, analyze and find patterns the patient is struggling with.

                Journal entry: {journal_entry}

                Tasks:
                1. What patterns can you discover?
                2. Give me the sentiment related to people occuring in the journal.
    """

    st.text("\nOutput:\n" + llm(prompt))
    return prompt

def ich_aktie(table):

    st.line_chart(table)

# create OpenAI instance

llm = OpenAI(
    temperature=0,
    openai_api_key=API_0,
    model_name = model
)

# get user input
#user_input = get_text()

# generate output
#if user_input:
 #   st.text(llm(user_input))

    
def main():
    
    user_input = get_text()
    if st.button("Send"):
        #st.text("Prompt: " + new_prompt + "\n\nPrompt for GPT: " + journal_analytics(user_input))
        st.text("This is the original prompt: " + journal_analytics(user_input))
    
    # call function to generate linechart
    if st.button("Generate chart"):
        st.text("This is the chart: ")
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['a', 'b', 'c'])
        ich_aktie(chart_data)

if __name__ == "__main__":
    main()



# TODO:
# chatbot anpassen
# jounrnal in langzeit speicher (langchain, pinecone, etc.)
# Analyse, auch grafisch
# tabelle