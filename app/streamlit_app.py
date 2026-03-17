import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

import streamlit as st
import streamlit.components.v1 as components
import whisper
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import random
import re
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Lecture Intelligence System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI Lecture Intelligence System")
st.caption("Upload a lecture audio file to get transcription, summary, topics, quiz, and analytics.")

# ─────────────────────────────────────────────
#  CACHE MODELS
# ─────────────────────────────────────────────

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

@st.cache_resource
def load_keybert():
    small_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return KeyBERT(model=small_model)

# ─────────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────────

def transcribe_audio(path):
    model  = load_whisper()
    result = model.transcribe(path, fp16=False)
    return result["text"]

@st.cache_data
def summarize_text(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    filtered  = [s for s in sentences if 60 < len(s) < 200]
    return " ".join(filtered[:3])

@st.cache_data
def extract_topics_with_scores(text):
    kw_model = load_keybert()
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=10
    )
    return keywords

@st.cache_data
def generate_mcqs(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    filtered  = [s for s in sentences if 40 < len(s) < 150]
    selected  = filtered[:5]

    mcqs = []
    for s in selected:
        words = re.findall(r'\b[A-Za-z]{5,}\b', s)
        if not words:
            continue
        key         = random.choice(words[:6])
        question    = s.replace(key, "_____")
        distractors = ["algorithm", "model", "dataset", "training", "prediction"]
        options     = random.sample(distractors, 3)
        options.append(key)
        random.shuffle(options)
        mcqs.append({"question": question, "options": options, "answer": key})

    return mcqs

# ─────────────────────────────────────────────
#  SAVE RESULTS
# ─────────────────────────────────────────────

def save_results(score, total, topics):
    result = {
        "score":      score,
        "total":      total,
        "percentage": round((score / total) * 100, 1) if total else 0,
        "topics":     topics,
        "timestamp":  str(datetime.now())
    }
    file = "outputs/results.json"
    os.makedirs("outputs", exist_ok=True)

    data = []
    if os.path.exists(file):
        try:
            with open(file, "r") as f:
                content = f.read().strip()
                data    = json.loads(content) if content else []
        except Exception:
            data = []

    data.append(result)
    with open(file, "w") as f:
        json.dump(data, f, indent=2)

# ─────────────────────────────────────────────
#  SESSION STATE DEFAULTS
# ─────────────────────────────────────────────

for key, default in {
    "processed_file":  None,
    "transcript":      "",
    "summary":         "",
    "topic_pairs":     [],
    "mcqs":            [],
    "start_quiz":      False,
    "quiz_submitted":  False,
    "user_answers":    [],
    "active_tab":      0,          # ← NEW: tracks which tab should be visible
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  QUIZ BUTTON CALLBACKS
# ─────────────────────────────────────────────

def start_quiz_callback():
    st.session_state.start_quiz     = True
    st.session_state.quiz_submitted  = False
    st.session_state.active_tab      = 3   # ← stay on Quiz tab (0-indexed)

def retake_quiz_callback():
    st.session_state.start_quiz     = False
    st.session_state.quiz_submitted  = False
    st.session_state.user_answers    = []
    st.session_state.active_tab      = 3   # ← stay on Quiz tab

# ─────────────────────────────────────────────
#  TAB PERSISTENCE HELPER
#  Injects a tiny JS snippet that re-clicks the
#  correct tab after every Streamlit rerun.
#  height=0 keeps it invisible.
# ─────────────────────────────────────────────

def keep_active_tab(tab_index: int):
    components.html(
        f"""
        <script>
        (function() {{
            var TARGET = {tab_index};

            function clickTab() {{
                var tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                if (tabs.length > TARGET) {{
                    tabs[TARGET].click();
                    return true;
                }}
                return false;
            }}

            // Try immediately (handles already-rendered tab bar)
            if (clickTab()) return;

            // Otherwise watch for the tab bar to appear in the DOM
            var observer = new MutationObserver(function(_, obs) {{
                if (clickTab()) {{
                    obs.disconnect();
                }}
            }});
            observer.observe(window.parent.document.body, {{
                childList: true,
                subtree: true
            }});

            // Safety net — disconnect after 3 s to avoid memory leak
            setTimeout(function() {{ observer.disconnect(); }}, 3000);
        }})();
        </script>
        """,
        height=0,
    )
# ─────────────────────────────────────────────
#  FILE UPLOAD
# ─────────────────────────────────────────────

st.header("📤 Upload Lecture Audio")

audio_file = st.file_uploader(
    "Supported formats: MP3, WAV, M4A",
    type=["mp3", "wav", "m4a"]
)

if audio_file is not None:

    file_path = f"uploaded_{audio_file.name}"
    with open(file_path, "wb") as f:
        f.write(audio_file.read())

    st.success(f"✅ **{audio_file.name}** uploaded successfully")

    if st.session_state.processed_file != audio_file.name:

        with st.spinner("🎙️ Transcribing audio with Whisper… this may take a minute"):
            st.session_state.transcript = transcribe_audio(file_path)

        with st.spinner("📝 Generating lecture summary…"):
            st.session_state.summary = summarize_text(st.session_state.transcript)

        with st.spinner("🔍 Extracting key topics…"):
            st.session_state.topic_pairs = extract_topics_with_scores(st.session_state.transcript)

        with st.spinner("❓ Generating quiz questions…"):
            st.session_state.mcqs = generate_mcqs(st.session_state.transcript)

        # Reset quiz + tab state for new file
        st.session_state.start_quiz      = False
        st.session_state.quiz_submitted   = False
        st.session_state.user_answers     = []
        st.session_state.processed_file   = audio_file.name
        st.session_state.active_tab       = 0   # back to Transcript on new upload

        st.success("🎉 Processing complete!")

    transcript   = st.session_state.transcript
    summary      = st.session_state.summary
    topic_pairs  = st.session_state.topic_pairs
    topic_words  = [w for w, s in topic_pairs]
    topic_scores = [round(s * 100, 1) for w, s in topic_pairs]
    mcqs         = st.session_state.mcqs

    # ── Lecture Stats Metric Cards ──────────────────────────
    word_count     = len(transcript.split())
    sentences      = [s for s in re.split(r'(?<=[.!?]) +', transcript) if s.strip()]
    sentence_count = len(sentences)
    reading_time   = max(1, round(word_count / 200))

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📝 Total Words",     word_count)
    c2.metric("📖 Sentences",       sentence_count)
    c3.metric("⏱️ Est. Read Time",  f"{reading_time} min")
    c4.metric("🔑 Topics Detected", len(topic_words))
    st.divider()

    # ─────────────────────────────────────────────
    #  TABS
    # ─────────────────────────────────────────────
# ─────────────────────────────────────────────
    #  CUSTOM TAB BAR  (no JS, no flicker)
    # ─────────────────────────────────────────────

    TAB_LABELS = ["📄 Transcript", "📝 Summary", "🔍 Topics", "❓ Quiz", "📊 Analytics"]

    # Render tab buttons
    cols = st.columns(len(TAB_LABELS))
    for i, label in enumerate(TAB_LABELS):
        if cols[i].button(
            label,
            key=f"tab_btn_{i}",
            use_container_width=True,
            type="primary" if st.session_state.active_tab == i else "secondary"
        ):
            st.session_state.active_tab = i

    st.divider()
    active = st.session_state.active_tab

    # ─────────────── TAB 1 : TRANSCRIPT ──────────────────
    if active == 0:
        st.subheader("Full Lecture Transcript")
        with st.expander("Show full transcript", expanded=False):
            st.write(transcript)

        st.subheader("📊 Sentence Length Distribution")
        lengths = [len(s.split()) for s in sentences]
        fig_hist = px.histogram(
            x=lengths, nbins=20,
            title="Words per Sentence",
            labels={"x": "Words per Sentence", "y": "Sentence Count"},
            color_discrete_sequence=["#4C78A8"]
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("🌐 Transcript Word Cloud")
        try:
            wc = WordCloud(
                width=900, height=400,
                background_color="white",
                colormap="Blues",
                max_words=100
            ).generate(transcript)
            fig_wc, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)
        except Exception as e:
            st.warning(f"Word cloud could not be generated: {e}")

    # ─────────────── TAB 2 : SUMMARY ─────────────────────
    elif active == 1:
        st.subheader("📋 Lecture Summary")
        if summary.strip():
            st.info(summary)
        else:
            st.warning("Summary could not be generated — transcript may be too short.")

        st.subheader("🔹 Key Points")
        key_sentences = [s for s in sentences if len(s.split()) > 10][:5]
        for i, s in enumerate(key_sentences, 1):
            st.markdown(f"**{i}.** {s}")

    # ─────────────── TAB 3 : TOPICS ──────────────────────
    elif active == 2:
        st.subheader("🔑 Detected Lecture Topics")

        if topic_words:
            topic_df = pd.DataFrame({
                "Topic":          topic_words,
                "Confidence (%)": topic_scores
            })

            fig_bar = px.bar(
                topic_df,
                x="Topic", y="Confidence (%)",
                text="Confidence (%)",
                title="Topic Relevance Confidence Scores",
                color="Confidence (%)",
                color_continuous_scale="Blues"
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("🕸️ Topic Coverage Radar")
            radar_fig = go.Figure(go.Scatterpolar(
                r=topic_scores + [topic_scores[0]],
                theta=topic_words + [topic_words[0]],
                fill="toself",
                line_color="royalblue",
                fillcolor="rgba(65, 105, 225, 0.2)"
            ))
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Topic Coverage Radar Chart"
            )
            st.plotly_chart(radar_fig, use_container_width=True)

            st.subheader("📋 Full Topic List")
            st.dataframe(
                topic_df.sort_values("Confidence (%)", ascending=False),
                use_container_width=True
            )
        else:
            st.warning("No topics could be extracted.")

    # ─────────────── TAB 4 : QUIZ ────────────────────────
    elif active == 3:
        if not st.session_state.start_quiz:
            st.info(f"📋 This quiz has **{len(mcqs)} questions** generated from the lecture.")
            st.button("▶️ Start Quiz", on_click=start_quiz_callback)

        elif not st.session_state.quiz_submitted:
            st.markdown("### Answer all questions, then click **Submit Quiz**.")

            for i, q in enumerate(mcqs):
                st.markdown(f"**Q{i+1}: {q['question']}**")
                st.radio(
                    "Choose your answer:",
                    q["options"],
                    key=f"quiz_ans_{i}",
                    index=None
                )
                st.divider()

            if st.button("✅ Submit Quiz"):
                st.session_state.user_answers = [
                    st.session_state.get(f"quiz_ans_{i}") for i in range(len(mcqs))
                ]
                st.session_state.quiz_submitted = True
                st.rerun()

        else:
            user_answers = st.session_state.get("user_answers", [])
            score        = 0
            wrong_topics = []
            result_data  = []

            for i, q in enumerate(mcqs):
                given   = user_answers[i] if i < len(user_answers) else None
                correct = given == q["answer"]
                if correct:
                    score += 1
                else:
                    for t in topic_words:
                        if t.lower() in q["question"].lower():
                            wrong_topics.append(t)

                result_data.append({
                    "Question":       f"Q{i+1}",
                    "Status":         "✅ Correct" if correct else "❌ Wrong",
                    "Your Answer":    given or "—",
                    "Correct Answer": q["answer"]
                })

            total      = len(mcqs)
            percentage = round((score / total) * 100, 1) if total else 0

            save_results(score, total, topic_words)

            st.subheader("🏆 Quiz Results")
            r1, r2, r3 = st.columns(3)
            r1.metric("Score",      f"{score} / {total}")
            r2.metric("Percentage", f"{percentage}%")
            r3.metric("Status",
                      "🌟 Excellent"   if percentage >= 80
                      else "👍 Good"   if percentage >= 60
                      else "📚 Needs Review")

            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=percentage,
                title={"text": "Understanding Level (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#2ecc71" if percentage >= 70
                                      else "#f39c12" if percentage >= 40
                                      else "#e74c3c"},
                    "steps": [
                        {"range": [0,  40], "color": "#ffcccc"},
                        {"range": [40, 70], "color": "#fff0b3"},
                        {"range": [70, 100], "color": "#ccffcc"}
                    ]
                }
            ))
            st.plotly_chart(gauge_fig, use_container_width=True)

            st.subheader("📋 Question-by-Question Breakdown")
            result_df = pd.DataFrame(result_data)
            st.dataframe(result_df, use_container_width=True)

            breakdown_fig = px.bar(
                result_df, x="Question", color="Status",
                title="Answer Status per Question",
                color_discrete_map={
                    "✅ Correct": "#2ecc71",
                    "❌ Wrong":   "#e74c3c"
                }
            )
            st.plotly_chart(breakdown_fig, use_container_width=True)

            if wrong_topics:
                st.warning("📚 **Topics to Revise:**")
                for t in list(set(wrong_topics)):
                    st.write(f"  🔸 {t}")
            else:
                st.success("🎉 Great job! You demonstrated good understanding of all topics.")

            st.button("🔄 Retake Quiz", on_click=retake_quiz_callback)

    # ─────────────── TAB 5 : ANALYTICS ───────────────────
    elif active == 4:
        st.subheader("📊 Learning Progress Analytics")

        file = "outputs/results.json"

        if os.path.exists(file):
            try:
                df = pd.read_json(file)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                df["Quiz #"] = df.index + 1

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Quizzes", len(df))
                m2.metric("Best Score",
                          f"{df['score'].max()} / {int(df['total'].max()) if 'total' in df.columns else '?'}")
                m3.metric("Average Score", f"{df['score'].mean():.1f}")
                m4.metric("Latest Score",  f"{df['score'].iloc[-1]}")

                st.divider()

                st.subheader("📈 Score Progress Over Time")
                fig_line = px.line(
                    df, x="timestamp", y="score",
                    markers=True,
                    title="Quiz Score Progress",
                    labels={"timestamp": "Date", "score": "Score"}
                )
                fig_line.update_traces(line_color="#4C78A8", marker_size=8)

                if len(df) >= 3:
                    df["moving_avg"] = df["score"].rolling(window=3).mean()
                    fig_line.add_trace(go.Scatter(
                        x=df["timestamp"], y=df["moving_avg"],
                        mode="lines", name="3-Quiz Moving Avg",
                        line=dict(dash="dash", color="orange", width=2)
                    ))

                st.plotly_chart(fig_line, use_container_width=True)

                if "percentage" in df.columns:
                    st.subheader("📉 Understanding % Trend")
                    fig_pct = px.area(
                        df, x="timestamp", y="percentage",
                        title="Understanding Level Over Time (%)",
                        labels={"percentage": "Score %", "timestamp": "Date"},
                        color_discrete_sequence=["#2ecc71"]
                    )
                    fig_pct.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig_pct, use_container_width=True)

                st.subheader("🥧 Topic Distribution")
                all_topics = []
                for t in df["topics"]:
                    if isinstance(t, list):
                        all_topics.extend(t)

                if all_topics:
                    topic_counts = pd.Series(all_topics).value_counts().reset_index()
                    topic_counts.columns = ["Topic", "Count"]

                    fig_pie = px.pie(
                        topic_counts,
                        names="Topic", values="Count",
                        hole=0.4,
                        title="Most Frequent Topics Across All Sessions"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    fig_tbar = px.bar(
                        topic_counts.head(10),
                        x="Topic", y="Count",
                        title="Top 10 Most Encountered Topics",
                        color="Count",
                        color_continuous_scale="Teal"
                    )
                    fig_tbar.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig_tbar, use_container_width=True)

                st.subheader("🗂️ Full Quiz History")
                cols_show = (["Quiz #", "timestamp", "score", "percentage"]
                        if "percentage" in df.columns
                        else ["Quiz #", "timestamp", "score"])
                display_df = df[cols_show].copy()
                display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(display_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error loading analytics: {e}")
        else:
            st.info("📭 No analytics data yet. Complete a quiz to start tracking your progress!")
