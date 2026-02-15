import os
import re
import io
import time
import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
import faiss

from docx import Document
from pypdf import PdfReader

from openai import OpenAI

st.set_page_config(page_title="ISPS Dashboard", layout="wide")

# -----------------------------
# OpenAI client (live suggestions)
# -----------------------------
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    client = OpenAI()
else:
    client = None

# -----------------------------
# File reading
# -----------------------------
def read_uploaded_file(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()

    if name.endswith(".txt") or name.endswith(".md"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs])

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n".join(pages)

    raise ValueError("Unsupported file type. Upload .txt/.md/.docx/.pdf")

# -----------------------------
# Parsers
# -----------------------------
def extract_actions(action_text: str) -> pd.DataFrame:
    """
    Prefer headings like:
      ### Action 1: ...
    Fallback:
      Action 1: ...
    """
    actions = []

    parts = re.split(r"^###\s+", action_text, flags=re.MULTILINE)
    if len(parts) > 5:
        for p in parts[1:]:
            lines = p.splitlines()
            title = (lines[0] if lines else "Untitled Action").strip()
            body = "\n".join(lines[1:]).strip()
            if title or body:
                actions.append({"action_title": title, "action_text": body})
        return pd.DataFrame(actions).head(300)

    # fallback for non-md
    lines = action_text.splitlines()
    cur_title, buf = None, []
    for line in lines:
        m = re.match(r"^(Action\s+\d+)\s*[:\-]\s*(.*)$", line.strip(), flags=re.IGNORECASE)
        if m:
            if cur_title and buf:
                actions.append({"action_title": cur_title, "action_text": "\n".join(buf).strip()})
            cur_title = f"{m.group(1)}: {m.group(2)}".strip()
            buf = []
        else:
            if cur_title is not None:
                buf.append(line)
    if cur_title and buf:
        actions.append({"action_title": cur_title, "action_text": "\n".join(buf).strip()})

    return pd.DataFrame(actions).head(300)

def normalize_strategic_objectives(strategic_text: str, n=10) -> pd.DataFrame:
    """
    Robust: build N objective units by chunking the objectives section,
    ensuring strategy-wise synchronization exists even if headings are messy.
    """
    m = re.search(r"(Strategic Objectives.*)", strategic_text, flags=re.IGNORECASE | re.DOTALL)
    core = m.group(1) if m else strategic_text
    core = re.sub(r"\n{3,}", "\n\n", core).strip()

    chunk_size = max(1500, len(core)//n)
    chunks, i = [], 0
    while i < len(core) and len(chunks) < n:
        chunks.append(core[i:i+chunk_size].strip())
        i += chunk_size
    while len(chunks) < n:
        chunks.append("")

    units = []
    for idx, ch in enumerate(chunks, start=1):
        units.append({
            "strategy_title": f"Strategic Objective {idx}: Objective {idx} (Normalized)",
            "strategy_text": ch if ch else "Details to be refined."
        })
    return pd.DataFrame(units)

# -----------------------------
# Embeddings + FAISS
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_alignment(strat_df: pd.DataFrame, act_df: pd.DataFrame, top_k=3):
    model = load_model()

    strategy_texts = [f"{r.strategy_title}\n{r.strategy_text}" for r in strat_df.itertuples(index=False)]
    action_texts   = [f"{r.action_title}\n{r.action_text}" for r in act_df.itertuples(index=False)]

    strat_emb = model.encode(strategy_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    act_emb   = model.encode(action_texts,   convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    dim = strat_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(strat_emb)

    scores, idxs = index.search(act_emb, top_k)

    rows = []
    for a_i, (sc_list, id_list) in enumerate(zip(scores, idxs)):
        for rank, (sc, s_i) in enumerate(zip(sc_list, id_list), start=1):
            rows.append({
                "action_id": a_i + 1,
                "action_title": act_df.iloc[a_i]["action_title"],
                "matched_strategy_rank": rank,
                "strategy_title": strat_df.iloc[s_i]["strategy_title"],
                "cosine_similarity": float(sc),
            })
    df = pd.DataFrame(rows)

    best = df[df["matched_strategy_rank"] == 1].copy()
    overall = float(best["cosine_similarity"].mean()) if len(best) else 0.0

    per_strategy = (
        best.groupby("strategy_title")["cosine_similarity"]
        .agg(actions_matched="count", mean_alignment="mean")
        .reset_index()
        .sort_values("mean_alignment", ascending=False)
    )

    return df, best, per_strategy, overall

# -----------------------------
# GPT suggestion
# -----------------------------
def gpt_suggestion(strategy_title: str, action_title: str, similarity: float) -> str:
    if not client:
        return "OPENAI_API_KEY not set. Add it in Streamlit Secrets to enable live suggestions."

    prompt = f"""
You are an expert strategy execution consultant.

Strategic Objective:
{strategy_title}

Current Action (low alignment; similarity={similarity:.3f}):
{action_title}

Task:
Improve the action so it clearly supports the objective.

Return in this exact structure:
1) Improved action (1–2 sentences)
2) Suggested KPI(s) (3–5 bullet points)
3) Missing tasks / steps (3–7 bullet points)
4) Timeline adjustment (1 sentence)
5) Owner/role suggestion (1 line)

Be practical, measurable, and aligned.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# UI
# -----------------------------
st.title("ISPS — Intelligent Strategic Plan Synchronization System")
st.caption("Upload Strategic + Action plans → FAISS similarity mapping → dashboard → live GPT improvements.")

with st.sidebar:
    st.header("Upload")
    strat_file = st.file_uploader("Strategic Plan (.md/.txt/.docx/.pdf)", type=["md","txt","docx","pdf"])
    act_file   = st.file_uploader("Action Plan (.md/.txt/.docx/.pdf)", type=["md","txt","docx","pdf"])

    st.header("Settings")
    top_k = st.slider("Top-K matches", 1, 5, 3)
    threshold = st.slider("Low alignment threshold", 0.20, 0.90, 0.60, 0.01)

    st.markdown("---")
    st.write("Live GPT:", "✅ Enabled" if client else "❌ Add OPENAI_API_KEY in Streamlit Secrets")

if not strat_file or not act_file:
    st.info("Upload both documents to start.")
    st.stop()

try:
    strategic_text = read_uploaded_file(strat_file)
    action_text = read_uploaded_file(act_file)
except Exception as e:
    st.error(f"File error: {e}")
    st.stop()

strat_df = normalize_strategic_objectives(strategic_text, n=10)
act_df = extract_actions(action_text)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Strategic Objectives (Normalized)")
    st.write("Units:", len(strat_df))
    st.dataframe(strat_df[["strategy_title"]], use_container_width=True)
with c2:
    st.subheader("Actions Detected")
    st.write("Actions:", len(act_df))
    st.dataframe(act_df[["action_title"]].head(30), use_container_width=True)

if st.button("Run Synchronization", type="primary"):
    with st.spinner("Running embeddings + FAISS matching..."):
        all_matches, best, per_strategy, overall = build_alignment(strat_df, act_df, top_k=top_k)

    m1, m2, m3 = st.columns(3)
    m1.metric("Overall Alignment (mean cosine)", f"{overall:.3f}")
    pct_good = float((best["cosine_similarity"] >= threshold).mean()) if len(best) else 0.0
    m2.metric(f"% Actions ≥ {threshold:.2f}", f"{pct_good*100:.1f}%")
    m3.metric("Total Actions", f"{len(act_df)}")

    st.subheader("Strategy-wise Synchronization")
    st.dataframe(per_strategy, use_container_width=True)

    st.subheader("Best Match per Action")
    view = best.copy()
    view["band"] = pd.cut(view["cosine_similarity"], bins=[-1, threshold, 0.75, 1.01], labels=["Poor", "Medium", "Good"])
    st.dataframe(view[["action_id","action_title","strategy_title","cosine_similarity","band"]],
                 use_container_width=True)

    st.download_button(
        "Download alignment_results.csv (Top-K matches)",
        data=all_matches.to_csv(index=False).encode("utf-8"),
        file_name="alignment_results.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download best_matches.csv (Top-1 match per action)",
        data=best.to_csv(index=False).encode("utf-8"),
        file_name="best_matches.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download strategy_alignment_summary.csv",
        data=per_strategy.to_csv(index=False).encode("utf-8"),
        file_name="strategy_alignment_summary.csv",
        mime="text/csv"
    )

    st.subheader("Live Intelligent Improvements (GPT)")
    low = best[best["cosine_similarity"] < threshold].sort_values("cosine_similarity").copy()

    st.write(f"Low-alignment actions (< {threshold:.2f}):", len(low))

    if len(low) > 0:
        pick = st.selectbox(
            "Select low-alignment action",
            options=low["action_id"].tolist(),
            format_func=lambda aid: f"Action {aid}: {low[low['action_id']==aid]['action_title'].values[0][:90]}"
        )
        row = low[low["action_id"] == pick].iloc[0]

        if st.button("Generate GPT Suggestion"):
            with st.spinner("Calling GPT..."):
                suggestion = gpt_suggestion(row["strategy_title"], row["action_title"], row["cosine_similarity"])
                st.text_area("Suggestion Output", suggestion, height=320)

    else:
        st.success("No low-alignment actions under current threshold.")
