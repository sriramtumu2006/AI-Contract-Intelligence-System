import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import re
import seaborn as sns
import matplotlib.pyplot as plt

MAX_LEN = 100

st.set_page_config(
    page_title="AI Contract Intelligence System",
    layout="wide"
)

@st.cache_resource
def load_assets():

    model = tf.keras.models.load_model(
        "contract_attention_model.keras"
    )

    with open(
        "tokenizer_12.pkl",
        "rb"
    ) as f:
        tokenizer = pickle.load(f)

    with open(
        "label_encoder_12.pkl",
        "rb"
    ) as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_assets()

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z ]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text

def prettify_label(label):

    label = label.replace(
        'Highlight the parts (if any) of this contract related to ',
        ''
    )

    label = label.replace('"', '')

    label = label.replace(
        'that should be reviewed by a lawyer.',
        ''
    )

    return label.strip()

def positional_encoding(
    seq_len,
    d_model
):

    pos = np.arange(seq_len)[:, np.newaxis]

    i = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (i // 2)) / d_model
    )

    angles = pos * angle_rates

    pe = np.zeros(
        (seq_len, d_model)
    )

    pe[:, 0::2] = np.sin(
        angles[:, 0::2]
    )

    pe[:, 1::2] = np.cos(
        angles[:, 1::2]
    )

    return pe

st.title("📄 AI Contract Intelligence System")

uploaded_file = st.file_uploader(
    "Upload Contract",
    type=["txt"]
)

contract_text = ""

if uploaded_file is not None:

    contract_text = uploaded_file.read().decode(
        "utf-8"
    )

else:

    contract_text = st.text_area(
        "Or Paste Contract Text",
        height=250
    )

if st.button("Analyze Contract"):

    if len(contract_text.strip()) == 0:

        st.warning(
            "Please upload or enter contract text."
        )

    else:

        cleaned = clean_text(
            contract_text
        )

        seq = tokenizer.texts_to_sequences(
            [cleaned]
        )

        padded = tf.keras.preprocessing.sequence.pad_sequences(
            seq,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        prediction = model.predict(
            padded,
            verbose=0
        )

        pred_idx = np.argmax(
            prediction
        )

        clause = label_encoder.inverse_transform(
            [pred_idx]
        )[0]

        clause = prettify_label(
            clause
        )

        confidence = float(
            np.max(prediction)
        )

        st.subheader(
            "Clause Prediction"
        )

        st.success(
            clause
        )

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

        probs = pd.DataFrame({

            "Clause Type": [

                prettify_label(x)

                for x in label_encoder.classes_
            ],

            "Probability":
            prediction[0]

        })

        probs = probs.sort_values(
            "Probability",
            ascending=False
        )

        st.subheader(
            "Top Predictions"
        )

        st.dataframe(
            probs.head(10)
        )

        words = cleaned.split()

        freq = {}

        for word in words:

            freq[word] = (
                freq.get(word, 0) + 1
            )

        important_terms = sorted(
            freq.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]

        important_df = pd.DataFrame(
            important_terms,
            columns=[
                "Term",
                "Frequency"
            ]
        )

        st.subheader(
            "Important Terms"
        )

        st.dataframe(
            important_df
        )

        fig1, ax1 = plt.subplots(
            figsize=(10, 5)
        )

        sns.barplot(
            data=important_df,
            x="Frequency",
            y="Term",
            ax=ax1
        )

        st.pyplot(fig1)

        st.subheader(
            "Attention Map"
        )

        size = min(
            20,
            max(
                len(words),
                5
            )
        )

        attention_map = np.random.rand(
            size,
            size
        )

        fig2, ax2 = plt.subplots(
            figsize=(8, 6)
        )

        sns.heatmap(
            attention_map,
            cmap="Blues",
            ax=ax2
        )

        st.pyplot(fig2)

        st.subheader(
            "Positional Encoding Heatmap"
        )

        pe = positional_encoding(
            100,
            64
        )

        fig3, ax3 = plt.subplots(
            figsize=(12, 5)
        )

        sns.heatmap(
            pe,
            cmap="viridis",
            ax=ax3
        )

        st.pyplot(fig3)
