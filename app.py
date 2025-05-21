import streamlit as st
import sounddevice as sd
import torchaudio
import tempfile
import torch
import re
import cv2
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ----------- Setup -------------
SAMPLE_RATE = 16000
MODEL_ID = "manandey/wav2vec2-large-xlsr-assamese"

# Get the base directory of app.py
BASE_DIR = Path(__file__).resolve().parent

# Dynamically define dataset path
DATASET_PATH = BASE_DIR / "datasets"

@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    return processor, model

processor, model = load_model()

# Assamese stopwords and dataset
assamese_stopwords = {
    "কিন্তু", "আছে", "এটা", "এটি", "এইটো", "তাত", "নেকি", "কৰি", "লাগে", "তেওঁ",
    "তুমি", "আমি", "মই", "আপুনি", "নাই", "তোমাক", "যি", "কোন", "তেওঁলোকে",
    "কৰা", "কৰিছা", "কৰিছে", "পাৰো", "পাৰিব", "ক'ৰ", "আহিছে", "দিয়া", "গৈ", "যাম"
}

dataset_sentences = [
    "অনুগ্ৰহ কৰি আপুনি এইটো পুনৰাবৃত্তি কৰিব পাৰিব নেকি", "অনুগ্ৰহ কৰি লাহে লাহে কথা পাতিব পাৰিব নেকি",
    "আজি আপুনি আজৰি আছে নিকি", "আপুনি ইমান দয়ালু", "আপুনি কি কৰিছে", "আপুনি কি কৰে", "আপুনি কি বিচাৰে",
    "আপুনি কি ভাবিছে", "আপুনি কিয় কান্দিছে", "আপুনি কিয় খং কৰি�ছে", "আপুনি কোন",
    "আপুনি কোনখন কলেজ_স্কুলৰ হয়", "আপুনি কৰিব পাৰিব", "আপুনি ক’ৰ পৰা আহিছে", "আপুনি খালেনে",
    "আপোনাক কেনেকৈ সহায় কৰিম", "আপোনাৰ কেৰিয়াৰৰ বাবে কি পৰিকল্পনা কৰিছে", "আপোনাৰ ফোন নম্বৰটো কি",
    "আপোনাৰ বয়স কিমান", "আমি বাহিৰলৈ যাম নেকি", "কি কলা তাক", "কি কৰিম মই দোমোজাত পৰিছো", "কি হ'ব বিচাৰিছা",
    "কি হ’ল", "কিবা লাগে নেকি", "কিবা লুকুৱাইছা নেকি", "কেতিয়া যাব ৰেলখন", "কেনেকৈ বিশ্বাস কৰিম তোমাক",
    "কেনেকৈ সাহস কৰিলে", "চিন্তা কৰাৰ প্ৰয়োজন নাই চিন্তা নকৰিব", "তাত মই আপোনাক সহায় কৰিব নোৱাৰো",
    "তুমি কিয় হতাশ হৈছা", "তুমি যিয়ে নকৰা কিয়, মই একো গুৰুত্ব নিদিওঁ।", "তেওঁ _তাই মোৰ বন্ধু",
    "তোমাৰ লগত কথা পাতি ভাল কাগিল", "নমস্কাৰ", "নমস্কাৰ, আপোনাৰ কি খবৰ", "মই আপোনাক সহায় কৰিব পাৰো নে",
    "মই একমত নহয়", "মই ঠিকেই আছো। ধন্যবাদ ছাৰ", "মোক কোনোবাই ৰখাই দিলে", "মোৰ তোমাক ভাল লাগে_মই তোমাক ভাল পাওঁ",
    "মোৰ বাবে ইয়াৰ কোনো পাৰ্থক্য নাই", "যোৱা আৰু শুই দিয়া", "সঁচা কথা কোৱা", "সি কোঠাটোত সোমাই গৈছে", "সি গৈ আছে"
]

# ----------- Functions -------------

def record_audio(duration=5):
    st.info("🎙️ Recording... Speak into your microphone")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.squeeze()

def transcribe(audio):
    input_values = processor(audio, return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.strip()

def preprocess(text):
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def gloss(text):
    tokens = preprocess(text).split()
    return [t for t in tokens if t not in assamese_stopwords]

def jaccard_similarity(tokens1, tokens2):
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 or not s2:
        return 0
    return len(s1 & s2) / len(s1 | s2)

def find_best_match(input_text):
    input_gloss = gloss(input_text)
    glossed_dataset = [gloss(sent) for sent in dataset_sentences]
    best_score, best_sentence = 0.0, None

    for sent, glossed in zip(dataset_sentences, glossed_dataset):
        score = jaccard_similarity(input_gloss, glossed)
        if score > best_score:
            best_score, best_sentence = score, sent

    if best_score >= 0.2:
        return best_sentence, best_score
    return None, 0.0  # No match found

def get_video_path(sentence):
    return DATASET_PATH / sentence / "ই.mp4"

# ----------- Streamlit UI -------------

st.title("🧏 Assamese Speech to Sign Language")
st.markdown("Speak or type in Assamese, and watch the matching sign language video.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input")
    text_input = st.text_area("Enter Assamese text:", height=100)

    if st.button("🎤 Record"):
        audio_data = record_audio(duration=5)
        transcription = transcribe(audio_data)
        st.success(f"📝 Transcribed Text: `{transcription}`")
        text_input = transcription  # Use transcribed text

    if text_input:
        best_sentence, score = find_best_match(text_input)
        if best_sentence:
            st.write(f"📘 Matched Sentence: `{best_sentence}` (Similarity: {score:.2f})")
        else:
            st.warning("⚠️ No matching sentence found. Please try a different input.")
    else:
        best_sentence = None

with col2:
    st.subheader("Sign Language Video")
    if best_sentence:
        video_path = get_video_path(best_sentence)
        if video_path.exists():
            st.video(str(video_path))
        else:
            st.error(f"⚠️ Video not found: `{video_path}`")
    else:
        st.info("No input provided or no match found. Please type or record to see the video.")
