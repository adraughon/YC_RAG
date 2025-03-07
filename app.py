import json
import numpy as np
import openai
from openai import OpenAI
import streamlit as st

openai_api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key = openai_api_key)

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Load the JSON data
with open('YC_Transcripts_Embedded.json', 'r') as f: # big file
    video_data = json.load(f)

# Function to calculate cosine similarity and retrieve relevant chunks
def retrieve_relevant_chunks(user_query, video_data, top_k=3):
    # Encode the user_query into an embedding
    query_embedding = encode_query(user_query)

    relevant_chunks = []

    for video in video_data:
        video_title = video["title"]
        for transcript in video["transcript"]:
            embedding = np.array(transcript["embedding"]).reshape(1, -1)
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding)
            relevant_chunks.append(
                (
                    similarity[0][0], 
                    f"**{video_title}**\nTimestamp: {transcript['timestamp']}\n\n{transcript['text_chunk']}\n"
                )
            )

    # Sort by similarity score and retrieve the top_k chunks
    relevant_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk[1] for chunk in relevant_chunks[:top_k]]


# Function to encode the query using OpenAI's text-embedding-3-small
def encode_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    # Extract the embedding from the response
    embedding = np.array(response.data[0].embedding)
    return embedding

st.title("ChatGPT-like clone")


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(prompt, video_data)
    raw_context = "\n".join(relevant_chunks)
    context = "The top 50 most popular YC YouTube videos have been vector embedded for RAG purposes. Here are the most relevant snippits as related to the user's query:\n" + raw_context

    with st.chat_message("user"):
        st.markdown(prompt)
        st.markdown('### RAG-Fetched Context:')
        st.markdown(context)

    # Create the assistant's prompt
    assistant_prompt = f"Context:\n{context}\nUser: {prompt}\nAssistant:"

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "user", "content": assistant_prompt}
            ],
            stream=True,
        )
        response = ""
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content"):
                if chunk.choices[0].delta.content is not None:
                    response += chunk.choices[0].delta.content
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
