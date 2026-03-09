import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

import streamlit as st
import whisper
from keybert import KeyBERT
import random
import re
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("AI Lecture Intelligence System")

# ---------- CACHE MODELS ----------

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

@st.cache_resource
def load_keybert():
    return KeyBERT()

@st.cache_data
def transcribe_audio(path):
    model = load_whisper()
    result = model.transcribe(path, fp16=False)
    return result["text"]

@st.cache_data
def summarize_text(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    filtered = [s for s in sentences if 60 < len(s) < 200]
    summary_sentences = filtered[:3]
    return " ".join(summary_sentences)

@st.cache_data
def extract_topics(text):
    kw_model = load_keybert()

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1,2),
        stop_words="english",
        top_n=10
    )

    return [w for w, s in keywords]

@st.cache_data
def generate_mcqs(text):

    sentences = re.split(r'(?<=[.!?]) +', text)
    filtered = [s for s in sentences if 40 < len(s) < 150]
    selected = filtered[:5]

    mcqs = []

    for s in selected:

        words = re.findall(r'\b[A-Za-z]{5,}\b', s)

        if len(words) == 0:
            continue

        key = random.choice(words[:6])
        question = s.replace(key, "_____")

        distractors = ["algorithm", "model", "dataset", "training", "prediction"]

        options = random.sample(distractors, 3)
        options.append(key)

        random.shuffle(options)

        mcqs.append({
            "question": question,
            "options": options,
            "answer": key
        })

    return mcqs


# ---------- SAVE RESULTS ----------

def save_results(score, topics):

    result = {
        "score": score,
        "topics": topics,
        "timestamp": str(datetime.now())
    }

    file = "outputs/results.json"

    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(file):
        data = []
    else:
        try:
            with open(file, "r") as f:
                content = f.read().strip()
                data = json.loads(content) if content else []
        except:
            data = []

    data.append(result)

    with open(file, "w") as f:
        json.dump(data, f, indent=2)


# ---------- UI ----------

st.header("Upload Lecture Audio")

audio_file = st.file_uploader(
    "Upload lecture audio",
    type=["mp3", "wav", "m4a"]
)

if audio_file is not None:

    file_path = "uploaded_lecture.mp3"

    with open(file_path, "wb") as f:
        f.write(audio_file.read())

    st.success("Lecture uploaded successfully")

    transcript = transcribe_audio(file_path)
    summary = summarize_text(transcript)
    topics = extract_topics(transcript)
    mcqs = generate_mcqs(transcript)

    # ---------- TABS ----------

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Transcript",
        "Summary",
        "Topics",
        "Quiz",
        "Analytics"
    ])

    # ---------- TRANSCRIPT ----------

    with tab1:

        st.subheader("Lecture Transcript")
        st.write(transcript[:1000])

    # ---------- SUMMARY ----------

    with tab2:

        st.subheader("Lecture Summary")
        st.write(summary)

    # ---------- TOPIC VISUALIZATION ----------

    with tab3:

        st.subheader("Detected Lecture Topics")

        topic_df = pd.DataFrame({
            "Topic": topics,
            "Importance": list(range(len(topics), 0, -1))
        })

        fig = px.bar(
            topic_df,
            x="Topic",
            y="Importance",
            text="Importance",
            title="Lecture Topic Importance"
        )

        fig.update_layout(transition_duration=600)

        st.plotly_chart(fig, use_container_width=True)

    # ---------- QUIZ ----------

    with tab4:

        if "start_quiz" not in st.session_state:
            st.session_state.start_quiz = False

        if not st.session_state.start_quiz:

            if st.button("Start Quiz"):
                st.session_state.start_quiz = True
                st.rerun()

        else:

            user_answers = []

            for i, q in enumerate(mcqs):

                st.write(f"Q{i+1}: {q['question']}")

                choice = st.radio(
                    "Choose answer",
                    q["options"],
                    key=f"quiz_{i}"
                )

                user_answers.append(choice)

            if st.button("Submit Quiz"):

                score = 0
                wrong_topics = []

                for i, q in enumerate(mcqs):

                    if user_answers[i] == q["answer"]:
                        score += 1
                    else:
                        for t in topics:
                            if t in q["question"]:
                                wrong_topics.append(t)

                st.subheader("Quiz Result")

                st.write(f"Score: {score} / {len(mcqs)}")

                save_results(score, topics)

                # ---------- GAUGE VISUALIZATION ----------

                percentage = (score / len(mcqs)) * 100

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=percentage,
                    title={'text': "Understanding Level"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'steps': [
                            {'range': [0, 40], 'color': "#ffcccc"},
                            {'range': [40, 70], 'color': "#fff0b3"},
                            {'range': [70, 100], 'color': "#ccffcc"}
                        ]
                    }
                ))

                st.plotly_chart(fig, use_container_width=True)

                if wrong_topics:

                    st.warning("You should revise these topics:")

                    weak = list(set(wrong_topics))

                    for t in weak:
                        st.write("-", t)

                else:

                    st.success("Good understanding of the lecture!")

    # ---------- ANALYTICS DASHBOARD ----------

    with tab5:

        st.subheader("Learning Progress Analytics")

        file = "outputs/results.json"

        if os.path.exists(file):

            df = pd.read_json(file)

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            fig = px.line(
                df,
                x="timestamp",
                y="score",
                markers=True,
                title="Quiz Score Progress Over Time"
            )

            fig.update_layout(transition_duration=800)

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Topic Distribution")

            all_topics = []

            for t in df["topics"]:
                all_topics.extend(t)

            topic_counts = pd.Series(all_topics).value_counts().reset_index()
            topic_counts.columns = ["Topic", "Count"]

            fig2 = px.pie(
                topic_counts,
                names="Topic",
                values="Count",
                hole=0.4,
                title="Most Frequent Lecture Topics"
            )

            st.plotly_chart(fig2, use_container_width=True)

        else:

            st.info("No analytics available yet. Take a quiz first.")