import streamlit as st
import numpy as np
import pickle
import json
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from beam_search import beam_search
from gtts import gTTS
import tempfile
import os

# BLIP

from transformers import BlipProcessor, BlipForConditionalGeneration

# BLEU

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt')

# ---------------- LOAD MODEL ----------------


def load_all():
    # 🔽 Download model from Google Drive (only once)
    if not os.path.exists("best_model.keras"):
        url = "https://drive.google.com/file/d/1SSHilapxY0OhUjeamkpgm3Caj5O7dEkR/view?usp=sharing"
        gdown.download(url, "best_model.keras", quiet=False)

    model = load_model("best_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("config.json") as f:
        config = json.load(f)

    base_model = EfficientNetB0(weights='imagenet', include_top=False)

    return model, tokenizer, config, base_model

model, tokenizer, config, base_model = load_all()
max_length = config["max_length"]

# ---------------- LOAD BLIP ----------------

from transformers import BlipProcessor, BlipForConditionalGeneration

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model
blip_processor, blip_model = load_blip()

# ---------------- LOAD CAPTIONS ----------------

def load_captions(file_path):
    mapping = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(',', 1)

        if len(parts) < 2:
            continue

        img, caption = parts
        img = img.strip()
        caption = caption.strip().lower()

        if img not in mapping:
            mapping[img] = []

        mapping[img].append(caption)

    return mapping


@st.cache_data
def get_caption_mapping():
    return load_captions("captions.txt")

caption_mapping = get_caption_mapping()

# ---------------- FEATURE EXTRACTION ----------------

def extract_features(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)


    feature = base_model.predict(img, verbose=0)
    feature = np.reshape(feature, (feature.shape[0], -1, feature.shape[-1]))

    return feature


# ---------------- GREEDY SEARCH ----------------

def greedy_search(photo):
    in_text = 'startseq'

    for _ in range(20):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding='post')

        preds = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(preds)

        word = tokenizer.index_word.get(yhat)
        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip()


# ---------------- HYBRID ----------------

def fast_caption(photo):
    caption = greedy_search(photo)


    weak_words = ["man", "woman", "something"]

    if any(w in caption for w in weak_words):
        caption = beam_search(model, tokenizer, photo, max_length, beam_width=5)

    return caption


# ---------------- BLIP ----------------

def blip_caption(image):
    inputs = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# ---------------- BLEU ----------------

def compute_bleu(references, candidate):
    smoothie = SmoothingFunction().method1
    references = [ref.split() for ref in references]
    candidate = candidate.split()
    return sentence_bleu(references, candidate, smoothing_function=smoothie)

# ---------------- NEUTRAL ----------------

def make_neutral(caption):
    caption = caption.lower()
    replace_map = {
    "man": "person",
    "woman": "person",
    "boy": "child",
    "girl": "child",
    "men": "people",
    "women": "people"
}
    for k, v in replace_map.items():
        caption = caption.replace(k, v)
    return caption

# ---------------- AUDIO ----------------

def text_to_speech(text):
    tts = gTTS(text=text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# ---------------- STREAMLIT UI ----------------
st.title("🖼️ Voice Based Image Captioning")

st.info("NOTE:- Upload image from Flickr30k dataset to see BLEU score")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ---------------- MAIN BLOCK ----------------

if uploaded_file is not None:


# Load image
    image = Image.open(uploaded_file).convert("RGB")
    img_name = uploaded_file.name

# Display image
    st.image(image, caption=img_name, width='stretch')

# Button MUST be inside
    if st.button("Generate Caption"):

        progress = st.progress(0)
        status = st.empty()

    # ---------------- STEP 1 ----------------
        status.text("Extracting features...")
        photo = extract_features(image)
        progress.progress(20)

    # ---------------- STEP 2 ----------------
        status.text("Generating caption (CNN + LSTM)...")
        caption = fast_caption(photo)
        caption = make_neutral(caption)
        progress.progress(50)

    # ---------------- STEP 3 ----------------
        status.text("Generating BLIP caption...")
        blip_cap = blip_caption(image)
        progress.progress(75)

    # ---------------- STEP 4 ----------------
        status.text("Computing BLEU score...")
        if img_name in caption_mapping:
            references = caption_mapping[img_name]

            bleu_model = compute_bleu(references, caption)
            bleu_blip = compute_bleu(references, blip_cap)
        else:
            bleu_model = None
            bleu_blip = None

        progress.progress(90)

    # ---------------- DONE ----------------
        status.text("Done!")
        progress.progress(100)

    # ---------------- RESULTS ----------------
        st.subheader("🧠 EfficientNetB0 Caption:")
        st.write(caption)

        audio1 = text_to_speech(caption)
        st.audio(audio1)

        st.subheader("🤖 BLIP Caption:")
        st.write(blip_cap)

        audio2 = text_to_speech(blip_cap)
        st.audio(audio2)

    # ---------------- BLEU ----------------
        if bleu_model is not None:
            st.subheader("📊 BLEU Scores")
            st.write(f"EfficientNetB0: {bleu_model:.4f}")
            st.write(f"BLIP: {bleu_blip:.4f}")
        else:
            st.warning("No reference captions found for this image.")

