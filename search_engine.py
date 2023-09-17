import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import faiss


# Create dataframe using csv file
def read_data(file):
    """
    Function that read csv file and create pandas dataframe and
    add an index column name as "ID" which will help to index text corpus using faiss algorithm
    
    Parameters
    ----------
    @param file path: str,
        A valid string specifying the path to the csv file which include corpus
        
    Returns
    -------
    @return pandas dataframe with additional index column
    """

    try:
        df = pd.read_csv(file, header=0, sep=';')
        df['ID'] = np.random.randint(0,100, size=len(df))
        return df
    
    except Exception as e:
        print(e)


# Function to transform and create sentense embeddings
def encode_corpus(df, model_name, col):
    """
    This function transform and create sentense embedding based on
    DistilBERT.
    
    Parameters
    ----------
    @param df: str,
        Corpus Dataframe which needs to be embedded
    @param model_name: str,
        Name of the DistilBERT model which use
    @param col: str,
        Name of the dataframe column which needs to be embedded
        
    Returns
    -------
    @return model: return sentense transformer model
    @return embeddings: embedded sentense list
    """

    try:
        # Instantiate the sentence-level DistilBERT
        model = SentenceTransformer(model_name)

        # Check if CUDA is available ans switch to GPU
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))

        print("Runs on CUDA: " + str(model.device))

        embeddings = model.encode(df[col].to_list(), show_progress_bar=True)

        return model, embeddings
    
    except Exception as e:
        print(e)


# Function to transform user query and find similar vectors
def search_vect(query, model, embeddings, df, num_results=2):
    """
    This function transform user query into same DistilBERT embeddings
    and find the similar vectors using FAISS algorithm.
    
    parameters
    ----------
    @param query: User input query
    @param model: Sentense transformer model
    @param embeddings Embedded corpus
    @param df: Corpus dataframe
    @param num_results: Number of results to return. Default 2
    
    Returns:
    -------
    @return D numpy array which includes distance between results and query
    @return I numpy array which includes ID of the results
    """

    try:
        # Change data type to float32
        embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

        # Instantiate the index
        index = faiss.IndexFlatL2(embeddings.shape[1])

        # Pass the index to IndexIDMap
        index = faiss.IndexIDMap(index)

        # Add vectors and their IDs
        index.add_with_ids(embeddings, df.ID.values)

        vector = model.encode(list(query))
        D, I = index.search(np.array(vector).astype("float32"), k=num_results)
        return D, I
    
    except Exception as e:
        print(e)


# Function to return top search results
def get_results(df, I, column):
    """
    Returns the Answer based on the sentense index
    """

    try:
        return [list(df[df.ID == idx][column]) for idx in I[0]]
    
    except Exception as e:
        print(e)