# AI-Coach

MVP web-app based on Streamlit

Folder "embeddings": consists of uploaded embeddings.
Folder "pages": all subpages for different features




- for the process we need an openai api key stored as environment variable locally. https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety

- "Embeddings local" or "Embeddings Pinecone": 
	A. The first step is to prepare the dataset for search by creating embeddings and save them either on Pinecone or locally as CSV-file.
		1. Upload, collect data
		2. Preprocess: split entries, chunk into short sections (Recursive splitting)
		3. Create Embeddings from each chunk
		4. Store in Pinecone or as CSV
	B. Search your journal
	


- The other variant is to use streamlit:
	0. Import all libraries needed
	1. Call "streamlit.run main.py" in the folder where the main-file is located
	2. Use the steps described on the streamlit page to create embeddings, search through the journal and also get feedback, textual and visual (altair)


- "GPT Prompt Testing":
	- No ML parts, only for testing different prompts and data extraction methods (based on GPT) on the journal

- "Feature Extraction Daily Entries": extract features from daily entries automatically and create matrix for long-term data visualization
	


"word-embedding-raw" and "doc-classification-bert-pytorch-raw" are similar, but manually calculated without paid tools. The models used are for example Word2Vec and BERT, therefore the results lack behind. But these models are easily interchangeable with state-of-the-art open source models from huggingface.
