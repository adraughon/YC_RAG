import json
import numpy as np
import openai
from openai import OpenAI
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

openai_api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=openai_api_key)

# Load the JSON data
with open('YC_Transcripts_Embedded.json', 'r') as f:  # big file
    video_data = json.load(f)

# Function to encode the query using OpenAI's text-embedding-3-small
def encode_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    # Extract the embedding from the response
    embedding = np.array(response.data[0].embedding)
    return embedding

# Function to retrieve relevant chunks and generate video links
def retrieve_relevant_chunks(user_query, video_data, top_k=3):
    query_embedding = encode_query(user_query)
    relevant_chunks = []

    for video in video_data:
        video_title = video["title"]
        video_url = video["url"]
        for transcript in video["transcript"]:
            embedding = np.array(transcript["embedding"]).reshape(1, -1)
            similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding)
            timestamp = transcript["timestamp"]
            timestamp_seconds = convert_timestamp_to_seconds(timestamp)
            
            # Construct embedded video URL
            embedded_url = f"https://www.youtube.com/embed/{video_url.split('v=')[1].split('&')[0]}?start={timestamp_seconds}"

            relevant_chunks.append(
                (
                    similarity[0][0],
                    embedded_url,
                    transcript['text_chunk'],
                    f"**{video_title}**\nTimestamp: {timestamp}\n\n{transcript['text_chunk']}\n"
                )
            )

    relevant_chunks.sort(key=lambda x: x[0], reverse=True)
    return relevant_chunks[:top_k]

# Convert timestamp format (hh:mm:ss or mm:ss) to total seconds
def convert_timestamp_to_seconds(timestamp):
    parts = list(map(int, timestamp.split(":")))
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return 0  # Default if format is unexpected

st.title(":orange[Yummy Cake] :cake:")
st.markdown("*Retrieval Augmented Generation (RAG) using the top 50 Y Combinator YouTube videos*")

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

    with st.chat_message("user"):
        st.markdown(prompt)
        st.markdown('### RAG-Fetched Context:')
        
        # Display video first, then the relevant context
        for _, video_url, just_transcript, chunk_text in relevant_chunks:
            # Embed the video with timestamp
            st.markdown(f'<iframe width="444" height="250" src="{video_url}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)
            with st.expander(just_transcript[:300] + " ... (click to show full snippet)"):
                st.markdown(chunk_text)
            st.markdown("")  # Add a separator for clarity

    # Create the assistant's prompt
    raw_context = "\n".join([chunk[2] for chunk in relevant_chunks])
    context = "The top 50 most popular YC YouTube videos have been vector embedded for RAG purposes. Here are the most relevant snippets as related to the user's query:\n\n" + raw_context

    assistant_prompt = f"Context:\n{context}\nUser: {prompt}\nAnswer the user's query like a partner at Y Combinator would"

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

# Waitlist Form (moved to the sidebar)
st.sidebar.subheader("Join our waitlist for a YC-minded AI-powered startup coach!\n Part-CRM, part-progress tracker, get after it you mad lad! :rocket:")

with st.sidebar.form(key='waitlist_form'):
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    submit_button = st.form_submit_button("Join Waitlist")

    if submit_button:
        # Process the submission (you could save the data to a file, database, etc.)
        st.sidebar.success(f"Thanks for signing up, {name}! We'll notify you once the tool is available!")
        # Here, you could add logic to save the data for further processing, e.g., appending to a waitlist file

