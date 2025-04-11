import streamlit as st
import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf
import soundfile as sf
import random
import os
import tempfile
import matplotlib.pyplot as plt
from fpdf import FPDF
# ---------- Custom Theme CSS ----------
# ---------- Custom Theme CSS ----------
st.markdown(
    """
    <style>
    body, .stApp { background-color: #f0f4fa; color: #0d47a1; font-family: 'Segoe UI', sans-serif; }
    h1, h2, h3, h4, h5, h6 { color: #0d47a1; font-weight: 600; }
    .stButton > button {
        background-color: #1e88e5 !important; color: white !important;
        border-radius: 10px; padding: 0.6rem 1.2rem; font-weight: 600;
        box-shadow: 0 4px 12px rgba(30,136,229,0.3); transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1565c0 !important; transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(21,101,192,0.4);
    }
    .plot-container { background-color: #fff; padding: 1rem; border-radius: 10px;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-top: 1.5rem; }
    .streamlit-expander { border: 1px solid #1e88e5; border-radius: 12px; background-color: #ffffff; }
    .streamlit-expander-header { color: #0d47a1; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #1e88e5 !important; border-radius: 20px; }
    footer, .css-qri22k {visibility: hidden;}

    
/* ===== Alerts Styling Fix - Universal Black Theme Attempt ===== */
    div[data-testid="stAlert"] {
        background-color: white !important;
        color: white !important;
        border-color: #0f5132 !important; /* To ensure borders don't look out of place */
    }

    div[data-testid="stAlert"] > div { /* Target the direct child div */
        background-color: #b9d4ff !important; //light blue bg
        color: white !important;
    }

    div[data-testid="stAlert"] * { /* Target all descendants */
        color: #003488 !important;
         font-weight: bold !important;
        
    }


    </style>
    """,
    unsafe_allow_html=True
)
  
# ---------- Title ----------
st.title("🗣️ Pronunciation Master")
st.markdown("👋 Let's test your pronunciation with **5 random sentences**.")

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pronunciation_model1.keras")
model = load_model()

# ---------- Load Sentences ----------
@st.cache_data
def load_sentences():
    with open("txt.done.data", "r", encoding='utf-8') as f:
        lines = f.readlines()
    return [line.split('"')[1] for line in lines if '"' in line]

sentences = load_sentences()
if not sentences:
    st.error("⚠️ No sentences found in txt.done.data")
    st.stop()

# ---------- Session State Setup ----------
if "random_sentences" not in st.session_state:
    st.session_state.random_sentences = random.sample(sentences, 5)
    st.session_state.current_index = 0
    st.session_state.scores = []
    st.session_state.recorded = False
    st.session_state.last_confidence = None

# ---------- Radar Chart ----------
def plot_radar_chart(scores_original):
    scores = scores_original.copy()
    labels = [f"Sentence {i+1}" for i in range(len(scores))]
    angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, scores, color='#1e88e5', linewidth=2)
    ax.fill(angles, scores, color='#bbdefb', alpha=0.5)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color="#1e88e5")
    ax.set_title("🎯 Pronunciation Radar Chart", size=16, color="#1e88e5", weight="bold", y=1.1)

    st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)



def generate_pdf(sentences, scores, avg_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    
    # Title
    pdf.set_font("DejaVu", size=16)
    pdf.set_text_color(30, 144, 255)
    pdf.cell(0, 10, txt="🗣️ Pronunciation Evaluation Report", ln=True, align='C')
    pdf.ln(10)

    # Sentences and scores
    pdf.set_font("DejaVu", size=12)
    pdf.set_text_color(0, 0, 0)

    for i, (sent, score) in enumerate(zip(sentences, scores), start=1):
        # Sentence
        pdf.multi_cell(0, 8, f"{i}. \"{sent.strip()}\"", align='L')
        # Score with alignment
        pdf.set_x(10)
        pdf.set_text_color(34, 139, 34)  # green
        pdf.cell(0, 8, f"   ➤ Score: {score:.2f}%", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    # Average score
    pdf.ln(5)
    pdf.set_font("DejaVu", size=13)
    pdf.set_text_color(30, 144, 255)
    pdf.cell(0, 10, txt=f"🌟 Average Score: {avg_score:.2f}%", ln=True)

    # Save
    path = "pronunciation_report.pdf"
    pdf.output(path)
    return path


# ---------- Main Flow ----------
index = st.session_state.current_index
if index < 5:
    st.subheader(f"📝 Sentence {index+1} of 5")
    st.info(f"👉 \"{st.session_state.random_sentences[index]}\"")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.recorded:
                if st.button("🎙️ Record"):
                    sr, duration = 22050, 5
                    st.warning("⏳ Recording... Speak clearly for 5 seconds.")
                    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
                    sd.wait()
                    st.success("✅ Recording done.")
                    path = os.path.join(tempfile.gettempdir(), f"user_sentence_{index}.wav")
                    sf.write(path, audio, sr)

                    y, sr = librosa.load(path, sr=22050)
                    if np.abs(y).max() < 0.01:
                        st.session_state.recorded = False
                        st.error("😶 Nothing recorded. Please speak clearly.")
                    else:
                        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                        mfccs_mean = np.mean(mfccs.T, axis=0)
                        input_data = mfccs_mean.reshape(1, 13, 1).astype(np.float32)
                        prediction = model.predict(input_data, verbose=0)
                        confidence = float(np.max(prediction)) * 100

                        st.session_state.last_confidence = confidence
                        st.session_state.recorded = True
                        st.progress(int(confidence), text=f"🎯 Accuracy: {confidence:.2f}%")

                        if confidence < 75:
                            st.warning("❌ Not accurate enough. Here's a tip: Slow down and articulate each syllable.")
                            st.markdown("- 📌 Try to emphasize vowel clarity.\n- 🧠 Think of the rhythm of native speakers.\n- 📢 Speak slightly louder and clearer.")  
                        else:
                            st.success("👍 Good pronunciation!")

            else:
                st.info(f"🎯 Accuracy: {st.session_state.last_confidence:.2f}%")

        with col2:
            if st.session_state.recorded:
                if st.button("✅ Next"):
                    st.session_state.scores.append(st.session_state.last_confidence)
                    st.session_state.current_index += 1
                    st.session_state.recorded = False
                    st.session_state.last_confidence = None
                    st.rerun()
                if st.button("🔁 Re-record"):
                    st.session_state.recorded = False
                    st.session_state.last_confidence = None
                    st.rerun()

else:
    st.header("🏆 Final Results")
    st.markdown("---")
    for i, score in enumerate(st.session_state.scores):
        st.markdown(f"**Sentence {i+1}:**")
        st.progress(int(score), text=f"{score:.2f}%")

    avg_score = np.mean(st.session_state.scores)
    st.subheader(f"📊 Overall Accuracy: **{avg_score:.2f}%**")

    with st.expander("🔍 Detailed Performance Radar Chart"):
        plot_radar_chart(st.session_state.scores)

    with st.expander("💬 Feedback"):
        if all(score >= 90 for score in st.session_state.scores):
            st.balloons()
            st.success("🎉 Fantastic! Your pronunciation is excellent!")
        elif avg_score >= 80:
            st.success("👍 Well done! Very good pronunciation.")
        elif avg_score >= 70:
            st.info("😊 Good effort! Keep practicing.")
        else:
            st.warning("Practice makes perfect! Focus on clear speech.")

    with st.expander("📄 Download Your Report"):
        path = generate_pdf(st.session_state.random_sentences, st.session_state.scores, avg_score)
        with open(path, "rb") as f:
            st.download_button("📥 Download Pronunciation Report", f, file_name="Pronunciation_Report.pdf")

    st.markdown("---")
    if st.button("🔄 Start Again"):
        for key in ["random_sentences", "current_index", "scores", "recorded", "last_confidence"]:
            st.session_state.pop(key, None)
        st.rerun()








# import streamlit as st
# import librosa
# import numpy as np
# import sounddevice as sd
# import tensorflow as tf
# import soundfile as sf
# import random
# import os
# import tempfile
# import matplotlib.pyplot as plt
# from fpdf import FPDF
# import base64


# <<<<css>>>>


# # # Page setup
# # st.set_page_config(page_title="🗣️ Pronunciation Master", layout="centered")
# # st.title("Interactive Pronunciation Evaluation Challenge")
# # st.markdown("👋 Pronounce **5 random sentences**. We'll score your clarity and consistency.")

# # Load model
# model = tf.keras.models.load_model("pronunciation_model1.keras")

# # Load sentences
# @st.cache_data
# def load_sentences():
#     with open("txt.done.data", "r", encoding='utf-8') as f:
#         lines = f.readlines()
#     return [line.split('"')[1] for line in lines if '"' in line]

# sentences = load_sentences()
# if not sentences:
#     st.error("⚠️ No sentences found in txt.done.data")
#     st.stop()

# # Session state
# if "random_sentences" not in st.session_state:
#     st.session_state.random_sentences = random.sample(sentences, 5)
#     st.session_state.current_index = 0
#     st.session_state.scores = []
#     st.session_state.recorded = False
#     st.session_state.last_confidence = None

# # Radar chart
# def plot_radar_chart(scores_original):
#     scores = scores_original.copy()
#     labels = [f"Sentence {i+1}" for i in range(len(scores))]

#     num_vars = len(scores)
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#     scores += scores[:1]
#     angles += angles[:1]

#     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
#     ax.plot(angles, scores, color='mediumvioletred', linewidth=2)
#     ax.fill(angles, scores, color='mediumvioletred', alpha=0.25)

#     ax.set_yticklabels([])
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, fontsize=10)
#     ax.set_title("🎯 Pronunciation Radar Chart", size=14, color="navy", weight="bold")

#     st.pyplot(fig)

# # PDF generation
# from fpdf import FPDF

# def generate_pdf(sentences, scores, avg_score):
#     pdf = FPDF()
#     pdf.add_page()
    
#     # Load Unicode font
#     pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
#     pdf.set_font("DejaVu", size=12)


#     pdf.set_text_color(33, 37, 41)
#     pdf.cell(200, 10, txt="🗣️ Pronunciation Evaluation Report", ln=True, align='C')

#     pdf.ln(10)
#     for i, (sent, score) in enumerate(zip(sentences, scores)):
#         pdf.multi_cell(0, 10, f"{i+1}. \"{sent}\"", align='L')
#         pdf.cell(0, 10, f"   ➤ Score: {score:.2f}%", ln=True)

#     pdf.ln(10)
#     pdf.set_font("DejaVu", '', 12)
#     pdf.cell(0, 10, txt=f"🌟 Average Score: {avg_score:.2f}%", ln=True)

#     pdf_path = "pronunciation_report.pdf"
#     pdf.output(pdf_path)
#     return pdf_path


# # Evaluation flow
# index = st.session_state.current_index
# if index < 5:
#     sentence = st.session_state.random_sentences[index]
#     st.subheader(f"📝 Sentence {index+1} of 5:")
#     st.info(f"👉 \"{sentence}\"")

#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("🎙️ Record"):
#             sr = 22050
#             duration = 5
#             st.warning("⏳ Recording... Speak clearly for 5 seconds.")
#             audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
#             sd.wait()
#             st.success("✅ Recording done.")

#             tmp_path = os.path.join(tempfile.gettempdir(), f"user_sentence_{index}.wav")
#             sf.write(tmp_path, audio, sr)

#             y, sr = librosa.load(tmp_path, sr=22050)
#             if np.abs(y).max() < 0.01:
#                 st.session_state.recorded = False
#                 st.error("😶 Nothing recorded. Please speak clearly into the mic.")
#             else:
#                 mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#                 mfccs_mean = np.mean(mfccs.T, axis=0)
#                 input_data = mfccs_mean.reshape(1, 13, 1).astype(np.float32)

#                 prediction = model.predict(input_data)
#                 confidence = float(np.max(prediction)) * 100

#                 st.session_state.last_confidence = confidence
#                 st.session_state.recorded = True
#                 st.progress(int(confidence), text=f"🎯 Pronunciation Accuracy: {confidence:.2f}%")

#                 if confidence < 75:
#                     st.warning("❌ Not accurate enough. Try re-recording or proceed to the next.")
#                 else:
#                     st.success("✅ That sounded great!")

#     with col2:
#         if st.session_state.recorded:
#             if st.button("✅ Next Sentence"):
#                 st.session_state.scores.append(st.session_state.last_confidence)
#                 st.session_state.current_index += 1
#                 st.session_state.recorded = False
#                 st.rerun()

#     if st.session_state.recorded:
#         st.markdown("---")
#         if st.button("🔁 Re-record"):
#             st.session_state.recorded = False
#             st.rerun()

# else:
#     st.header("📊 Final Evaluation Results")
#     st.markdown("---")

#     for i, score in enumerate(st.session_state.scores):
#         st.markdown(f"**Sentence {i+1}:**")
#         st.progress(int(score), text=f"{score:.2f}%")

#     st.markdown("---")
#     avg_score = np.mean(st.session_state.scores)
#     st.subheader("🌟 Average Pronunciation Accuracy")
#     st.progress(int(avg_score))
#     st.markdown(f"### 🔥 **{avg_score:.2f}%**")

#     plot_radar_chart(st.session_state.scores)

#     # Feedback
#     if all(score >= 90 for score in st.session_state.scores):
#         st.balloons()
#         st.success("🏆 Legendary! You're a native-like speaker!")
#     elif avg_score >= 80:
#         st.success("🎉 Great job! You're improving fast.")
#     else:
#         st.warning("📈 Keep practicing! You're on your way.")

#     # PDF download

#     pdf_path = generate_pdf(st.session_state.random_sentences, st.session_state.scores, avg_score)
#     with open(pdf_path, "rb") as f:
#         st.download_button(
#             label="📄 Download Report as PDF",
#             data=f,
#             file_name="Pronunciation_Report.pdf",
#             mime="application/pdf"
#         )


#     # Try again
#     if st.button("🔁 Try Again"):
#         for key in ["random_sentences", "current_index", "scores", "recorded", "last_confidence"]:
#             if key in st.session_state:
#                 del st.session_state[key]
#         st.rerun()


# import streamlit as st
# import librosa
# import numpy as np
# import tensorflow as tf

# st.title("🗣️ Pronunciation Evaluation App")
# st.markdown("Upload your recorded speech and get feedback on your pronunciation.")

# # Load model
# model = tf.keras.models.load_model("pronunciation_model1.keras")

# uploaded_audio = st.file_uploader("Upload a WAV audio file", type=["wav"])

# if uploaded_audio is not None:
#     st.audio(uploaded_audio, format="audio/wav")
    
#     # Load audio
#     y, sr = librosa.load(uploaded_audio, sr=16000)
    
#     # Step 1: Extract Mel spectrogram with 13 bins
#     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13, hop_length=160)  # hop=160 → ~10ms stride
    
#     # Step 2: Convert to decibel scale
#     mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
#     # Step 3: Resize or pad to (13, 100)
#     if mel_db.shape[1] < 100:
#         pad_width = 100 - mel_db.shape[1]
#         mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
#     else:
#         mel_db = mel_db[:, :100]
    
#     # Step 4: Normalize (optional but good)
#     mel_db = (mel_db - np.min(mel_db)) / (np.max(mel_db) - np.min(mel_db))
    
#     # Step 5: Reshape to (1, 13, 100, 1)
#     input_data = mel_db.reshape(1, 13, 100, 1).astype(np.float32)

#     # Prediction
#     prediction = model.predict(input_data)
#     confidence = float(np.max(prediction)) * 100
#     label = np.argmax(prediction)

#     st.success(f"✅ Pronunciation Score: {confidence:.2f}%")
#     st.info(f"🔠 Predicted Label: {label}")







































# import streamlit as st
# import librosa
# import numpy as np
# import tensorflow as tf
# import io

# # Load your model
# model = tf.keras.models.load_model("pronunciation_checker_model.keras")  # Replace with actual path

# st.title("🗣️ Pronunciation Evaluation App")
# st.markdown("Upload your recorded speech and get feedback on your pronunciation.")

# # Upload audio file
# uploaded_audio = st.file_uploader("Upload a WAV audio file", type=["wav"])

# if uploaded_audio is not None:
#     # Display the audio player
#     st.write("Here is your uploaded audio:")
#     st.audio(uploaded_audio, format='audio/wav')

#     # Load audio data for processing
#     y, sr = librosa.load(uploaded_audio, sr=16000)

#     # Preprocess: Convert to Mel-spectrogram (you can tweak this as per your model)
#     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#     mel_spec_db = mel_spec_db[..., np.newaxis]  # Add channel dimension

#     # Resize or pad the input if required by model input shape
#     input_shape = (128, 128, 1)  # Example shape
#     mel_spec_resized = tf.image.resize(mel_spec_db, input_shape[:2])
#     mel_spec_resized = tf.expand_dims(mel_spec_resized, axis=0)

#     # Predict
#     prediction = model.predict(mel_spec_resized)
#     confidence = float(np.max(prediction)) * 100
#     label = np.argmax(prediction)

#     # Show results
#     st.success(f"✅ Pronunciation Score: {confidence:.2f}%")
#     st.info(f"🔠 Predicted Label: {label}")







# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import numpy as np
# import librosa  # For audio processing (MFCC, spectrogram, etc.)
# import io       # To handle audio bytes

# # Load your Keras model
# try:
#     model = load_model("pronunciation_checker_model.keras")
# except Exception as e:
#     st.error(f"Error loading the model: {e}")
#     st.stop()

# # --- AUDIO PREPROCESSING FUNCTION (ADAPT THIS!) ---
# def preprocess_audio(audio_bytes):
#     try:
#         # Load the audio data using librosa
#         audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=None) # Keep original sampling rate

#         # --- EXTRACT AUDIO FEATURES (MATCH YOUR MODEL'S TRAINING) ---
#         # Example using MFCCs:
#         mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=20) # Adjust n_mfcc
#         processed_audio = np.transpose(mfccs, axes=[1, 0]) # Time-major

#         # --- PADDING/TRUNCATING (IF YOUR MODEL EXPECTS FIXED LENGTH) ---
#         # Example: Pad to a maximum length
#         max_len = 100 # Adjust based on your model's input shape
#         current_len = processed_audio.shape[0]
#         if current_len < max_len:
#             padding = np.zeros((max_len - current_len, processed_audio.shape[1]))
#             processed_audio = np.vstack((processed_audio, padding))
#         elif current_len > max_len:
#             processed_audio = processed_audio[:max_len, :]

#         # Reshape for the model (add batch dimension)
#         processed_audio = np.expand_dims(processed_audio, axis=0)

#         return processed_audio

#     except Exception as e:
#         st.error(f"Error preprocessing audio: {e}")
#         return None

# # --- PRONUNCIATION CHECKING FUNCTION ---
# def check_pronunciation_audio(model, processed_audio):
#     if processed_audio is not None:
#         prediction = model.predict(processed_audio)
#         # --- INTERPRET PREDICTION (ADAPT BASED ON YOUR MODEL'S OUTPUT) ---
#         # This is a placeholder - you need to implement the actual interpretation
#         if np.mean(prediction) > 0.5:
#             result = "Pronunciation might be okay."
#         else:
#             result = "Pronunciation might need improvement."
#         return result, prediction
#     else:
#         return "Could not process audio.", None

# # --- STREAMLIT APPLICATION ---
# st.title("Audio Pronunciation Checker")
# st.subheader("Record audio to check pronunciation (basic example)")

# audio_bytes = st.audio(label="Record your pronunciation:", format="audio/wav")

# if audio_bytes:
#     if model is not None:
#         with st.spinner("Processing audio and checking pronunciation..."):
#             processed_audio = preprocess_audio(audio_bytes)
#             if processed_audio is not None:
#                 result, raw_prediction = check_pronunciation_audio(model, processed_audio)
#                 st.subheader("Result:")
#                 st.write(result)

#                 if st.checkbox("Show raw prediction"):
#                     st.write("Raw Prediction:")
#                     st.write(raw_prediction)
#             else:
#                 st.error("Audio preprocessing failed.")
#     else:
#         st.error("Model not loaded.")

# st.info("This is a basic example. You'll need to implement audio preprocessing (feature extraction) that matches how your model was trained and carefully interpret the model's output.")