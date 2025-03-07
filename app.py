import json
import numpy as np
from openai import OpenAI
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Load the JSON data
with open('path_to_your_file.json', 'r') as f:
    video_data = json.load(f)

# Function to calculate cosine similarity and retrieve relevant chunks
def retrieve_relevant_chunks(user_query, video_data, top_k=3):
    # You would need to encode the user_query into an embedding as well.
    query_embedding = encode_query(user_query)  # Define your encoding method
    
    relevant_chunks = []
    
    for video in video_data:
        for transcript in video['transcript']:
            embedding = np.array(transcript['embedding']).reshape(1, -1)
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding)
            relevant_chunks.append((similarity[0][0], transcript['text_chunk']))
    
    # Sort by similarity score and retrieve the top_k chunks
    relevant_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk[1] for chunk in relevant_chunks[:top_k]]

# Placeholder encoding function for the query - replace with actual embedding method
def encode_query(query):
    # This function should transform the query into the same embedding space as the transcripts
    # You can use OpenAI embeddings API or any other method you prefer
    # Here we return a dummy zero vector for illustration
    return np.zeros(768)  # Adjust size according to your embeddings

st.title("ChatGPT-like clone")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(prompt, video_data)
    context = "\n".join(relevant_chunks)

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
        response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
