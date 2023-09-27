import streamlit as st
from langchain.llms import OpenAI
import pandas as pd
import numpy as np
from datetime import datetime
import os
import re


# get user input
# split journal-data into entries (check if prompt + entry is < max_tokens)
# call gpt-3 for each entry, save output (sentiment, summary etc.) in a dictionary/array
# then call gpt-3 for a analysis of the dictionary (the whole journal)
# recommend a course of action (e.g. therapy, meditation, etc.)



#------------------#
# 0. streamlit settings/page configuration
st.set_page_config(page_title="GPT-3 Chatbot", page_icon=":rocket:", layout="wide")
st.sidebar.markdown("#  Main page")
st.title("AI-Coach")
st.text("1. For an analysis of your journal entries, please upload your journal text or write it in the text area below. To get a different analysis, either change the model or the sourcecode of main.py")
st.text("2. For a chart of your journal entries, please select the timeframe and click on 'Generate chart'")
st.text("3. In the section 'page chatbot' you can chat with the AI-Coach. To start type in your OpenAI API key and select a model. Then click on 'Start chatbot'")
st.text("4. In the section 'page embeddings' you can upload a text file and get the embeddings of the text. Afterwards you can ask questions and chat with your journal")

st.sidebar.header("Instuctions")
st.sidebar.info("This is a demo of an AI-Coach. You can upload your journal here and get feedback. For other functions please select the corresponding tab on the left")
API_0 = os.getenv('OPENAI_API_KEY')
model = st.sidebar.selectbox(label= 'Model', options=["gpt-3.5-turbo", "text-davinci-003", "text-ada-001"])
st.sidebar.info("Select a model according your task.")
#------------------#



# function to call gpt and generate analysis
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

    return llm(prompt)


# function to call gpt and generate analysis of one day
def daily_analytics(journal_entry, past_entry):
    # journal_entry: string of journal entry of one day
    # return: values for datamatrix
    prompt = f"""
            Rate the sentiment of a journal text between 0 (negative) and 10 (positive). Rate in relation to the sentiment and summary of the day before.
            Rating of the day before ("float, string (summary)"): {past_entry}

            Text: {journal_entry}
            
            Give me the answer in the following format back: "day; float (sentiment); string (summary of the entry in 1-2 sentences)"
            """
    return llm2(prompt)


# graph function, table as input
def ich_aktie(table):
    st.line_chart(table)


# create OpenAI instances
## standard instance
llm = OpenAI(
    temperature=0,
    openai_api_key=API_0,
    model_name = model
)
## instance for daily analytics, higher temperature, limited max_tokens
llm2 = OpenAI(
    temperature=0.5,
    max_tokens=30,
    openai_api_key=API_0,
    model_name = model
)




# split text into journal entries
def split_on_empty_lines(s):
    blank_line_regex = r"(?:\r?\n){2,}"
    return re.split(blank_line_regex, s.strip())



    
def main():
    #------------------#
    # 1. Get user input (text or file-upload) and call function to generate output
    ## text input
    user_input = st.text_area("Please import your Journal entries here as plain text: ", placeholder="Your journal text here.")
    if st.button("Send"):
        st.text("Output: " + journal_analytics(user_input))
    
    
    ## upload file & call function to split into journal entries
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
    if uploaded_file is not None:
        file = uploaded_file.read().decode("utf-8")            
        #st.write(split_on_empty_lines(file))

        data_summary = []

        for day in split_on_empty_lines(file):
            if not data_summary: 
                data_summary.append(daily_analytics(day, ""))
                continue

            data_summary.append(daily_analytics(day, data_summary[-1]))
            
        st.write(data_summary)

    # todo: check if entry is < max_tokens. Check how much text is possible
    # .split(';') um die einzelnen Werte zu trennen. Entweder 3 Variablen als return Wert oder loop über data_summary.


    ## use local file instead 
    with open('Journal-short.txt') as f:
        lines = f.readlines()
        #for line in lines:
            #if line.strip():
                #st.write(line)
        #st.write(lines)

    #------------------#
    


    # slider for datetime for chart
    timeframe = st.slider("Select timeframe of Journal entries",
                          value=(datetime(2021, 1, 1), datetime(2021, 1, 31)),
                            format="DD/MM/YYYY")
    #st.write("Select start and end date: ", timeframe)


    # call function to generate linechart
    if st.button("Generate chart"):
        st.text("This is the chart: ")
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['a', 'b', 'c'])
        ich_aktie(chart_data)



    

if __name__ == "__main__":
    main()


