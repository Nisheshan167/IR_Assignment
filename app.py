import os
import re
import io
import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
import faiss

from docx import Document
from pypdf import PdfReader

# Optional LLM suggestions
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    from openai import OpenAI
    _client = OpenAI()

st.set_page_config(page_title="ISPS – Plan Synchronization", layout="wide")

# -----------------------------
# Helpers: file reading
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
        # PDF extraction can be imperfect, but we support it
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n".join(pages)

    raise ValueError("Unsupported file type. Upload .txt/.md/.docx/.pdf")


# -----------------------------
# Helpers: parsing
# -----------------------------
def extract_strategic_units(strategic_text: str) -> pd.DataFrame:
    """
    Prefer headings like:
      ## Strategic Objective 1: ...
    Fallback: chunk whole doc
    """
    lines = strategic_text.splitlines()
    pattern = r"^##\s+(.*)$"

    units = []
    current_title = None
    buf = []

    for line in lines:
        m = re.match(pattern, line.strip())
        if m:
            if current_title and buf:
                units.append({"strategy_title": current_title.strip(),
                              "strategy_text": "\n".join(buf).strip()})
            current_title = m.group(1)
            buf = []
        else:
            if current_title is not None:
                buf.append(line)

    if current_title and buf:
        units.append({"strategy_title": current_title.strip(),
                      "strategy_text": "\n".join(buf).strip()})

    # fallback if too few headings
    if len(units) < 5:
        chunk_size = 2500
        chunks = [strategic_text[i:i+chunk_size] for i in range(0, len(strategic_text), chunk_size)]
        units = [{"strategy_title": f"Strategic Chunk {i+1}", "strategy_text": c.strip()}
                 for i, c in enumerate(chunks[:12]) if c.strip()]

    return pd.DataFrame(units)


def extract_actions(action_text: str) -> pd.DataFrame:
    """
    Prefer headings like:
      ### Action 1: ...
    Fallback: bullet points
    """
    parts = re.split(r"^###\s+", action_text, flags=re.MULTILINE)
    actions = []

    if len(parts) > 5:
        for p in parts[1:]:
            lines = p.splitlines()
            title = (lines[0] if lines else "Untitled Action").strip()
            body = "\n".join(lines[1:]).strip()
            if title or body:
                actions.append({"action_title": title, "action_text": body})
    else:
        bullets = re.split(r"^\-\s+", action_text, flags=re.MULTILINE)
        for b in bullets[1:]:
            lines = b.splitlines()
            title = (lines[0] if lines else "Untitled Action").strip()[:120]
            body = "\n".join(lines[1:]).strip()
            actions.append({"action_title": title, "action_text": (title + "\n" + body).strip()})

    df = pd.DataFrame(actions)
    # cap to keep runtime safe
    return df.head(250)


def chunk_text(text: str, max_chars=1200, overlap=150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        end = min(len(text), i + max_chars)
        chunks.append(text[i:end])
        i = end - overlap
        if i < 0:
            i = 0
        if end == len(text):
            break
    return chunks


# -----------------------------
# Core: Alignment engine
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")  # fast, good enough

def build_alignment(strat_df: pd.DataFrame, act_df: pd.DataFrame, top_k=3):
    model = load_model()

    strategy_texts = [
        f"{r.strategy_title}\n{r.strategy_text}" for r in strat_df.itertuples(index=False)
    ]
    action_texts = [
        f"{r.action_title}\n{r.action_text}" for r in act_df.itertuples(index=False)
    ]

    # Embed (normalize so cosine = inner product)
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
# Optional: Suggestions
# -----------------------------
def suggest_improvements(strategy_title: str, action_title: str, similarity: float) -> dict:
    """
    If OPENAI_API_KEY exists -> GPT suggestion
    else -> template fallback
    """
    if USE_OPENAI:
        prompt = f"""
You are improving an organization's Action Plan to better align with its Strategic Plan.

Strategy objective:
{strategy_title}

Action:
{action_title}

Similarity score: {similarity:.3f} (low)

Return:
1) Improved action wording (1-2 sentences)
2) Suggested KPI(s) (2-4 bullet points)
3) Missing tasks (3-6 bullet points)
4) Timeline adjustment suggestion (1 sentence)

Be practical, measurable, and aligned.
"""
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user", "content": prompt}],
            temperature=0.4,
        )
        txt = resp.choices[0].message.content.strip()
        return {"suggestion_text": txt, "source": "OpenAI GPT"}

    # Fallback (no API key)
    return {
        "suggestion_text": (
            f"Improved action: Reframe the action to explicitly deliver outcomes tied to '{strategy_title}'.\n\n"
            "Suggested KPIs:\n"
            "- % completion by quarter\n"
            "- Outcome metric directly related to the strategy\n"
            "- Quality/accuracy/uptime metric (if digital)\n\n"
            "Missing tasks:\n"
            "- Define owner + milestones\n"
            "- Add baseline measurement + target\n"
            "- Add stakeholder sign-off / governance checkpoint\n"
            "- Add risks + mitigations\n\n"
            "Timeline: Break into phased delivery (pilot → rollout → optimize)."
        ),
        "source": "Template (no API key)"
    }


# -----------------------------
# UI
# -----------------------------
st.title("ISPS — Intelligent Strategic Plan Synchronization System")
st.caption("Upload a Strategic Plan and Action Plan → alignment scoring (FAISS + embeddings) → dashboard + recommendations.")

with st.sidebar:
    st.header("Inputs")
    strat_file = st.file_uploader("Upload Strategic Plan (.md/.txt/.docx/.pdf)", type=["md","txt","docx","pdf"])
    act_file   = st.file_uploader("Upload Action Plan (.md/.txt/.docx/.pdf)", type=["md","txt","docx","pdf"])

    st.header("Settings")
    top_k = st.slider("Top-K strategy matches per action", 1, 5, 3)
    threshold = st.slider("Low-alignment threshold", 0.20, 0.90, 0.60, 0.01)
    st.markdown("---")
    st.write("LLM Suggestions:", "✅ Enabled" if USE_OPENAI else "⚠️ Template only (set OPENAI_API_KEY to enable GPT)")

if not strat_file or not act_file:
    st.info("Upload both files to begin.")
    st.stop()

try:
    strategic_text = read_uploaded_file(strat_file)
    action_text = read_uploaded_file(act_file)
except Exception as e:
    st.error(f"File reading error: {e}")
    st.stop()

strat_df = extract_strategic_units(strategic_text)
act_df = extract_actions(action_text)

colA, colB = st.columns(2)
with colA:
    st.subheader("Detected Strategic Units")
    st.write(f"{len(strat_df)} units found")
    st.dataframe(strat_df[["strategy_title"]].head(20), use_container_width=True)
with colB:
    st.subheader("Detected Actions")
    st.write(f"{len(act_df)} actions found")
    st.dataframe(act_df[["action_title"]].head(20), use_container_width=True)

run = st.button("Run Synchronization", type="primary")

if not run:
    st.stop()

with st.spinner("Computing embeddings + FAISS matching..."):
    all_matches, best_matches, per_strategy, overall = build_alignment(strat_df, act_df, top_k=top_k)

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Overall Alignment (mean best-match cosine)", f"{overall:.3f}")
pct_good = float((best_matches["cosine_similarity"] >= threshold).mean()) if len(best_matches) else 0.0
m2.metric("% Actions ≥ Threshold", f"{pct_good*100:.1f}%")
m3.metric("Total Actions", f"{len(act_df)}")

st.subheader("Strategy-wise Synchronization")
st.dataframe(per_strategy, use_container_width=True)

st.subheader("Action → Best Strategy Mapping")
best_display = best_matches.copy()
best_display["alignment_band"] = pd.cut(
    best_display["cosine_similarity"],
    bins=[-1, threshold, 0.75, 1.01],
    labels=["Poor", "Medium", "Good"]
)
st.dataframe(best_display[["action_id","action_title","strategy_title","cosine_similarity","alignment_band"]],
             use_container_width=True)

# Download outputs
st.download_button(
    "Download alignment_results.csv (all top-K matches)",
    data=all_matches.to_csv(index=False).encode("utf-8"),
    file_name="alignment_results.csv",
    mime="text/csv"
)
st.download_button(
    "Download best_matches.csv (top-1 match per action)",
    data=best_matches.to_csv(index=False).encode("utf-8"),
    file_name="best_matches.csv",
    mime="text/csv"
)

# Suggestions
st.subheader("Intelligent Improvement Suggestions")
low = best_matches[best_matches["cosine_similarity"] < threshold].copy()
st.write(f"Low-alignment actions (< {threshold:.2f}): {len(low)}")

if len(low) == 0:
    st.success("No low-alignment actions under current threshold.")
else:
    pick = st.selectbox(
        "Select a low-alignment action",
        options=low["action_id"].tolist(),
        format_func=lambda aid: f"Action {aid}: {low[low['action_id']==aid]['action_title'].values[0][:80]}"
    )
    row = low[low["action_id"] == pick].iloc[0]
    if st.button("Generate Suggestion (LLM / Template)"):
        out = suggest_improvements(row["strategy_title"], row["action_title"], row["cosine_similarity"])
        st.info(f"Source: {out['source']}")
        st.text_area("Suggestion Output", out["suggestion_text"], height=260)

# Evaluation (simple)
st.subheader("Testing & Evaluation (Optional)")
st.caption("Upload a small ground-truth CSV with columns: action_id, true_strategy_title. We compute precision@1 at threshold and overall accuracy-like stats.")
gt_file = st.file_uploader("Upload Ground Truth CSV", type=["csv"], key="gt")
if gt_file:
    gt = pd.read_csv(gt_file)
    merged = best_matches.merge(gt, on="action_id", how="inner")
    merged["correct"] = merged["strategy_title"].str.strip().str.lower() == merged["true_strategy_title"].str.strip().str.lower()
    precision_at1 = merged["correct"].mean() if len(merged) else 0.0
    above = merged[merged["cosine_similarity"] >= threshold]
    precision_at1_above = above["correct"].mean() if len(above) else 0.0
    st.write("Rows compared:", len(merged))
    st.metric("Precision@1 (all compared)", f"{precision_at1*100:.1f}%")
    st.metric(f"Precision@1 (only ≥ {threshold:.2f})", f"{precision_at1_above*100:.1f}%")
    st.dataframe(merged[["action_id","action_title","strategy_title","true_strategy_title","cosine_similarity","correct"]],
                 use_container_width=True)
