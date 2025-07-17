# Prototype Chatbot mit SocialMap Daten

Dieses Projekt ist ein Prototyp eines Chatbots, der Informationen aus der **Social Map Berlin** einbindet. Er basiert auf **Streamlit**, nutzt **OpenAI Embeddings** für semantische Suche und ermöglicht eine Art **einfaches RAG (Retrieval-Augmented Generation)**.

## 🚀 Features

- Streamlit-App mit einfachem User Interface
- Lädt Social Map Berlin Daten aus einer öffentlichen JSON-Quelle
- Erstellt Embeddings für die Social Map-Einträge
- Vergleicht Nutzereingaben semantisch mit diesen Daten
- Generiert Antworten basierend auf den relevantesten Treffern und GPT-4o-mini
- Caching von Daten & Embeddings für schnelleren Zugriff

## 🛠️ Setup

### Voraussetzungen

- Python 3.10 (idealerweise in Conda-Umgebung)
- OpenAI API Key

### Installation

1. **Repository klonen:**
```bash
git clone https://github.com/DEIN_USERNAME/DEIN_REPO.git
cd DEIN_REPO
Environment aktivieren (optional, wenn Conda):

bash
Kopieren
Bearbeiten
conda activate openai_310
Benötigte Libraries installieren:

bash
Kopieren
Bearbeiten
pip install -r requirements.txt
API-Key konfigurieren:
Lege eine Datei .streamlit/secrets.toml an:

toml
Kopieren
Bearbeiten
OPENAI_API_KEY = "sk-..."
Start der App
bash
Kopieren
Bearbeiten
streamlit run Prototype_Chat_SocialMap.py
📊 Datenquelle
Die Daten stammen von der öffentlichen Social Map API:

arduino
Kopieren
Bearbeiten
https://public.socialmap-berlin.de/items
⚡ Weiterentwicklungsideen
Fortschrittsanzeige bei Embedding-Generierung

Treffer-Visualisierung vor der Antwort

Speicherung von Chatverläufen

Deployment via Streamlit Cloud oder Docker

🧑‍💻 Autor
Raffael Ruppert