# AI-Coach

MVP web-app based on Streamlit


! for the process we need an openai api key stored as environment variable locally. https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety


- "Embeddings local" or "Embeddings Pinecone" to **chat with your journal**:
  	A. The first step is to prepare the dataset for search by creating embeddings and save them either on Pinecone or locally as CSV-file.
		1. Upload, collect data
		2. Preprocess: split entries, chunk into short sections (Recursive splitting)
		3. Create Embeddings from each chunk
		4. Store in Pinecone or as CSV
	B. Search your journal
	
- The other variant is to use streamlit ("ai_coach"-folder):
	0. Import all libraries needed
	1. Call "streamlit.run main.py" in the terminal in the folder where the main-file is located
	2. Use the steps described on the streamlit page to create embeddings, search through the journal and also get feedback, textual and visual (altair)
	
	The folder "embeddings" consists of uploaded and created embeddings. The folder "pages" contains all streamlit subpages for different features.

- "GPT Prompt Testing":
	- No ML parts, only for testing different prompts and data extraction methods (based on GPT) on the journal

- "Feature Extraction Daily Entries": extract features from daily entries automatically and create matrix for long-term data visualization
	

- "NLP": First manual drafts for embeddings and classification with models like Word2Vec and BERT, but not on journal data. Very basic models but also very easy interchangeable in the future with state of the art methods.

