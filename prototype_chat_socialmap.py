import numpy as np
import requests
import streamlit as st
import os
import pickle
from datetime import datetime
from openai import OpenAI

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

st.set_page_config(page_title="Social Map Chatbot", layout="wide")

st.title("Prototyp: Social Map Chatbot")

@st.cache_data
def lade_json_daten():
    response = requests.get("https://public.socialmap-berlin.de/items")
    return response.json()

daten = lade_json_daten()

def erzeuge_embeddings(daten, _client, batch_size=20):
    texte = []
    embeddings = []

    items = daten["items"]
    all_texts = []
    for eintrag in items:
        titel = eintrag.get("title", "")
        beschreibung = eintrag.get("description", {}).get("de", "")
        text = titel + "\n" + beschreibung
        texte.append(text)
        all_texts.append(text)

    progress_bar = st.progress(0)
    total = len(all_texts)

    for i in range(0, total, batch_size):
        batch_texts = all_texts[i:i+batch_size]
        response = _client.embeddings.create(
            model="text-embedding-ada-002",
            input=batch_texts
        )
        for data in response.data:
            embeddings.append(data.embedding)

        progress_bar.progress(min((i + batch_size) / total, 1.0))

    progress_bar.empty()
    return texte, np.array(embeddings)

def lade_oder_erzeuge_embeddings(force_neu=False):
    cache_file = "embeddings_cache.pkl"
    timestamp_file = "embeddings_timestamp.txt"

    if not force_neu and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            texte, embeddings = pickle.load(f)
        with open(timestamp_file, "r") as f:
            timestamp = f.read()
        return texte, embeddings, timestamp

    texte, embeddings = erzeuge_embeddings(daten, client)

    with open(cache_file, "wb") as f:
        pickle.dump((texte, embeddings), f)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(timestamp_file, "w") as f:
        f.write(timestamp)

    return texte, embeddings, timestamp

def finde_relevante_texte(prompt, texte, embeddings, client, top_k=3):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[prompt]
    )
    query_embedding = np.array(response.data[0].embedding)

    ähnlichkeiten = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    beste_indices = np.argsort(ähnlichkeiten)[-top_k:][::-1]
    relevante_texte = [texte[i] for i in beste_indices]
    return relevante_texte

# UI Layout
col1, col2 = st.columns([1, 3], gap="large")

with col1:
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px;'>
        <h3>Info & Steuerung</h3>
        <p><b>Über den Chatbot:</b></p>
        <ul>
            <li>Antwortet auf Basis der Social Map Berlin.</li>
            <li>Setzt OpenAI-Embeddings für semantische Suche ein.</li>
            <li>Antworten basieren nur auf den Social Map-Daten.</li>
        </ul>
        <p>Es handelt sich um einen <b>Prototypen</b>. Feedback gerne an <a href='mailto:raffael.ruppert@sciencespo.fr'>Raffael Ruppert</a>.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Embeddings aktualisieren"):
        texte, embeddings, timestamp = lade_oder_erzeuge_embeddings(force_neu=True)
        st.success(f"Embeddings wurden neu erstellt am: {timestamp}.")
    else:
        texte, embeddings, timestamp = lade_oder_erzeuge_embeddings()

    st.info(f"Letzte Embedding-Aktualisierung: {timestamp}")

with col2:
    st.markdown("""
    <div style='background-color: #f7f9fc; padding: 10px; border-radius: 8px;'>
        <h3>Chat</h3>
    </div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_area("Schreibe deine Nachricht:", height=100)

    if st.button("Absenden"):
        if user_input:
            relevante_infos = finde_relevante_texte(user_input, texte, embeddings, client)

            root_prompt = (
                "Du bist ein hilfsbereiter, präziser und verständlicher Chatbot, spezialisiert auf die Informationen der Social Map Berlin.\n"
                "Nutze ausschließlich die bereitgestellten Kontextinformationen, um die Nutzerfrage zu beantworten.\n"
                "Wenn du keine passende Information findest, erkläre dies höflich und verweise darauf, dass nur die Social Map-Daten verwendet werden.\n"
                "Antworten sollen sachlich, freundlich und in einer klaren Sprache formuliert sein.\n"
                "Die Originalstruktur des Datensatzes beinhaltet folgende Spalten: title, image, state, tags, primaryTopic, location, address, zip, city, latitude, longitude, responsible, website, email, contact, phone, facebook, lastEditDate, mobile, proposalFor, resubmissionDate, resubmissionNotification, twitter, whatsapp, apiKeyUsed, instagram, location_ref, projectEndDate, projectStartDate, telegram, vimeo, youtube, id, brief.de, brief.en, description.de, description.en, hours.de, hours.en, proposals, sponsors.\n"
                "Wenn du über ein Angebot sprichst, schau ob du einen passenden Link finden kannst."
            )

            system_prompt = root_prompt + "\n\nKontextinformationen:\n"
            for info in relevante_infos:
                system_prompt += f"- {info}\n"

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
            )
            antwort = completion.choices[0].message.content

            st.session_state.chat_history.append(("User", user_input))
            st.session_state.chat_history.append(("Bot", antwort))

    for role, message in reversed(st.session_state.chat_history):
        if role == "User":
            st.markdown(f"<div style='background-color:#e1f5fe; padding:10px; border-radius:5px; margin:5px 0'><b>Du:</b> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#ede7f6; padding:10px; border-radius:5px; margin:5px 0'><b>Chatbot:</b> {message}</div>", unsafe_allow_html=True)
