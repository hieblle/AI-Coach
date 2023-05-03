import pandas as pd
import numpy as np
import streamlit as st
import openai
import os
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from openai.embeddings_utils import get_embedding, cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter



# define openai api key and embedding model
openai.api_key = os.getenv('OPENAI_API_KEY')
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

#------------------#
# 0. streamlit settings/page configuration
st.set_page_config(page_title="Ask your Journal!", page_icon=":sparkles:", layout="wide")
st.sidebar.markdown("# Work with Embeddings")
st.title("Text Embeddings Demo")

st.sidebar.header("Instuctions")
st.sidebar.info("1. Upload your journal entries \n 2. Search for a term \n 3. Get the most similar text parts")
st.sidebar.info("To import existing embeddings, please name your file 'Journal_embedding.csv' and place it in the 'pages' folder, then refresh the page.")
#------------------#


# create embeddings once (from journal-short.xlsx)
def create_embeddings_from_local():
    df = pd.read_excel("journal-short.xlsx")
    df['embedding'] = df['journal-short'].apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
    df.to_csv('Journal_embedding.csv')

def create_embeddings_from_upload(df):
    df['embedding'] = df['journal-short'].apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
    df.to_csv('pages/Journal_embedding.csv')

def create_embeddings_from_txt_upload(df):
    df['embedding'] = df['Text'].apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
    df.to_csv('pages/Journal_embedding.csv')


def split_text(content):
    """Split content into entries based on date pattern and return a Pandas DataFrame.
       content: string with journal entries
       df: Pandas DataFrame with dates and entries"""
    # Define a regular expression pattern to match dates
    date_pattern = r'\d{4}.\d{2}.\d{2}'

    # Split the content based on the date pattern
    entries = re.split(date_pattern, content)

    # Extract dates from the content
    dates = re.findall(date_pattern, content)

    # Create a dictionary with dates and corresponding entries
    data = {'Date': dates, 'Text': [entry.strip() for entry in entries[1:]]}

    # Create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    return df


def langchain_textsplitter(content_entry):
    """Split entries into chunks.
       content_entry: string with one journal entry
       text_array: list of strings with chunks of the entry"""
    
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    )
    text_array = []
    texts = text_splitter.create_documents([content_entry])
    for chunk in texts:
        text_array.append(chunk.page_content)

    return text_array


# function to search in embeddings
def search(df, search_term, n=3):
    """ 
    df: dataframe with embeddings
    search_term: string to search for
    n: number of results to return
    """

    # convert embeddings to numpy array
    df["embedding"] = df["embedding"].apply(eval).apply(np.array)

    # get embedding of search term
    search_embeddings = get_embedding(search_term, engine="text-embedding-ada-002")

    # calculate cosine similarity
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, search_embeddings))

    # sort by similarity and return top n
    return df.sort_values("similarity", ascending=False).head(n)


def plot_embeddings(df):
    # extract embeddings from dataframe, convert to numpy array
    #embeddings = df["embedding"].apply(eval).apply(np.array).to_numpy()
    embeddings = df['embedding'].to_numpy()
    st.write(embeddings.shape)

    # Perform dimensionality reduction
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)

    # plot embeddings
    fig, ax = plt.subplots()
    ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], marker='o', alpha=0.2)
    ax.xlabel("PCA 1")
    ax.ylabel("PCA 2")
    ax.title("2D Text Embeddings using PCA")
    st.pyplot(fig)

    #todo: try local without streamlit, jupyter

    


if __name__ == "__main__":
    ## manually upload file from local machine and create embeddings
    #create_embeddings_from_local()

    # upload journal entries, create embeddings
    # check if embeddings already exist
    path = "pages/Journal_embedding.csv"
    if os.path.exists(path):
        st.write('You have already created your embeddings. You can search them now.')
    else:
        st.write("Upload your journal entries here as an excel file (.xlsx). (Each line represents one journal entry.)")
        uploaded_file = st.file_uploader("Choose an Excel-file")
        if uploaded_file is not None:
            dataframe = pd.read_excel(uploaded_file)
            create_embeddings_from_upload(dataframe)
            st.success("Your embeddings have been created. You can search them now.")
        st.write("Choose a .txt file as alternative:")
        uploaded_file = st.file_uploader("Choose a .txt-file", type="txt")
        if uploaded_file is not None:
            file = uploaded_file.read().decode("utf-8")
            df = split_text(file)   # split file into entrys, returns pd.Dataframe
            chunked_df = pd.DataFrame(columns=['Date', 'Text'])
            # iterate over df (entries), chunked_df: pd.Dataframe with chunks
            for index, row in df.iterrows():
                chunks = langchain_textsplitter(row['Text']) # split text into chunks, return: list of strings
                date_vector = [row['Date']]*len(chunks)
                chunked_df = pd.concat([chunked_df, pd.DataFrame({'Date': date_vector, 'Text': chunks})], ignore_index = True)
       
            create_embeddings_from_txt_upload(chunked_df)
            st.success("Your embeddings have been created. You can search them now.")
            st.write(chunked_df)



    #filename = st.text_input(label="How is your file called? Or how do you want to call it?", placeholder="Example: Journal_entries.csv")
    if os.path.exists(path):

        # search embeddings (user defines search term, read embeddings from csv, search, return results)
        search_term = st.text_input(label="Search term", placeholder="Enter search term here")
        search_button = st.button(label="Search", type="primary")
        data = pd.read_csv('pages/Journal_embedding.csv')
        if search_button:
            answer = search(data, search_term)
            for index, row in answer.iterrows():
                st.write(row["similarity"], row["Text"])

        if st.button('Extract Embeddings'):
            plot_embeddings(data)
            st.write('Done!')


# streamlit: 
# - let user create embeddings from their journal entries, define name for csv
#       - how to change data format? We want txt files to upload. 
# - save them in pinecone for later use
# - button for restoring saved embeddings (look up in pinecone or local)