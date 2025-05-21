import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
import tempfile
import os

st.title("Assamese Speech to Sign Language Translator")

# Load pre-trained model (replace with your Assamese model if available)
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

processor, model = load_model()

# Set up streamlit-webrtc for recording
st.header("Record Your Voice")
webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
    audio_receiver_size=1024,
)

# Audio buffer to collect chunks
audio_frames = []

class AudioProcessor:
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert audio frame to numpy and store it
        audio = frame.to_ndarray().flatten()
        audio_frames.append(audio)
        return frame

webrtc_ctx.audio_receiver.set_processor(AudioProcessor())

# Button to stop and transcribe
if st.button("Transcribe Speech"):
    if len(audio_frames) == 0:
        st.warning("Please record something first.")
    else:
        with st.spinner("Processing..."):
            # Concatenate all recorded frames
            audio_data = np.concatenate(audio_frames, axis=0).astype(np.float32)

            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                torchaudio.save(f.name, torch.tensor(audio_data).unsqueeze(0), 16000)
                file_path = f.name

            # Load and resample if needed
            waveform, sample_rate = torchaudio.load(file_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            input_values = processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).input_values
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])

            st.success("Transcription:")
            st.write(transcription)

            # You can now apply glossing, matching, and play the sign video
            # For example:
            # glossed_text = gloss(transcription)
            # matched_video = match_gloss(glossed_text)
            # st.video(matched_video)

            # Clean up
            os.remove(file_path)

# Display instructions
st.info("Click 'Start' to record audio from your browser, then click 'Transcribe Speech' to process it.")
