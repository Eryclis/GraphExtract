import pandas as pd
import numpy as np
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import random
import sys
# First experiment - load only one document

from langchain_community.document_loaders import TextLoader
splitter = RecursiveCharacterTextSplitter(
    chunk_size=4500,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

from df_helpers import *


import uuid
import pandas as pd
import numpy as np

import requests
import json

#pip install -q -U google-generativeai

import google.generativeai as genai


def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk.page_content,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df

def graphPrompt3(input: str, metadata={}):
    
    GOOGLE_API_KEY=''

    genai.configure(api_key=GOOGLE_API_KEY)
    
    prompt = f'''
        **Instructions:**

        You are a network graph maker who extracts entities and relationships from a provided text (input) and use them to build a Knowledge Graph.

        1. Carefully read the provided text to identify mentioned entities. Entities can be people, places, organizations, events, concepts, etc.Only consider the name of the most semantically relevant identified entities. Ignore the referencies/citations along the text.

        2. Identify concise, meaningfull relationships between the mentioned entities in the text. Relationships can be actions, associations, properties, etc. Only consider the most semantically relevant relationships between entities.

        3. Based on the identified entities and relationships, construct a Knowledge Graph representing entities as nodes and relationships as edges.

        4. Take care of your output token limit. Be concise in your output and strictly comply with the expected output structure (list of JSON objects).
        
        **Example:**
        
        *Context (input):*
            "Albert Einstein was born in Ulm, in the Kingdom of Württemberg, in the German Empire, on March 14, 1879. He was a theoretical physicist, best known for developing the theory of relativity (Silva et al., Rodrigues et al), one of the two pillars of modern physics."

        *Expected Output (list of JSON objects):*
            [
                {{"node_1": "Albert Einstein", "node_2": "Ulm", "edge": "born in"}},
                {{"node_1": "Ulm", "node_2": "Kingdom of Württemberg", "edge": "located in"}},
                {{"node_1": "Kingdom of Württemberg", "node_2": "German Empire", "edge": "located in"}},
                {{"node_1": "Albert Einstein", "node_2": "March 14, 1879", "edge": "date of"}},
                {{"node_1": "Albert Einstein", "node_2": "Theory of relativity", "edge": "known for"}},
                {{"node_1": "Theory of relativity", "node_2": "Modern physics", "edge": "one of the two pillars of"}}
            ]

        *Context:*
            "{input}"

        *Expected Output (list of JSON objects):*

    '''
    
    model = genai.GenerativeModel()
    response = model.generate_content(prompt)
    try:
        result = json.loads(response.text)
        result = [dict(item) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
        
    return result

def df2Graph(dataframe: pd.DataFrame, model=None) -> list:
    # dataframe.reset_index(inplace=True)
    results = dataframe.apply(
        lambda row: graphPrompt3(row.text, {"chunk_id": row.chunk_id}), axis=1
    )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list

def graph2Df(nodes_list) -> pd.DataFrame:
    ## Remove all NaN entities
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())

    return graph_dataframe


path="semantic/"
files = os.listdir(path)
start = int(sys.argv[1])
num =int(sys.argv[2])
for f in files[start:start+num]:
    loader = TextLoader(path+f, encoding="utf-8")
    load_doc = loader.load()
    texts = splitter.split_documents(load_doc)
    print(texts[0].page_content[:10])
    df = documents2Dataframe(texts)
    df_copy = df.copy()
    concepts_list = df2Graph(df_copy)
    dfg1 = graph2Df(concepts_list)
    dfg1.replace("", np.nan, inplace=True)
    dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
    dfg1.to_csv("out/"+f,sep="|", index=False)

