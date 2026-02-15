import os, re, io
import numpy as np
import pandas as pd
import streamlit as st

from docx import Document
from pypdf import PdfReader

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from openai import OpenAI

st.set_page_config(page_title="ISPS Dashboard", layout="wide")

# -----------------------------
# OpenAI
# -----------------------------
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# -----------------------------
# Session state init
# -----------------------------
def init_state():
    defaults = {
        "sync_done": False,
        "all_matches": None,
        "best": None,
        "per_strategy": None,
        "overall": 0.0,
        "emb_label": "",
        "gpt_text": "",
        "strat_bytes": None,
        "act_bytes": None,
        "strat_name": None,
        "act_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# -----------------------------
# File reading (from bytes)
# -----------------------------
def read_file_from_bytes(filename: str, data: bytes) -> str:
    name = filename.lower()

    if name.endswith(".txt") or name.endswith(".md"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs])

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(pages)

    raise ValueError("Unsupported file type. Upload .txt/.md/.docx/.pdf")

# -----------------------------
# Parsing
# -----------------------------
def extract_actions(action_text: str) -> pd.DataFrame:
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

    # fallback: "Action N: ..."
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
    m = re.search(r"(Strategic Objectives.*)", strategic_text, flags=re.IGNORECASE | re.DOTALL)
    core = m.group(1) if m else strategic_text
    core = re.sub(r"\n{3,}", "\n\n", core).strip()

    chunk_size = max(1500, len(core)//n) if n else max(1500, len(core))
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
# Embeddings: TF-IDF only (cloud-safe, no downloads)
# -----------------------------
def embed_joint(strategy_texts, action_texts):
    joint = strategy_texts + action_texts
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    joint_emb = vectorizer.fit_transform(joint).toarray().astype(np.float32)

    norm = np.linalg.norm(joint_emb, axis=1, keepdims=True) + 1e-9
    joint_emb = joint_emb / norm

    strat_emb = joint_emb[:len(strategy_texts)]
    act_emb   = joint_emb[len(strategy_texts):]
    return strat_emb, act_emb, "TF-IDF"

# -----------------------------
# Alignment
# -----------------------------
def build_alignment(strat_df: pd.DataFrame, act_df: pd.DataFrame, top_k=3):
    strategy_texts = [f"{r.strategy_title}\n{r.strategy_text}" for r in strat_df.itertuples(index=False)]
    action_texts   = [f"{r.action_title}\n{r.action_text}" for r in act_df.itertuples(index=False)]

    strat_emb, act_emb, emb_label = embed_joint(strategy_texts, action_texts)
    sim = cosine_similarity(act_emb, strat_emb)

    rows = []
    for i in range(sim.shape[0]):
        order = np.argsort(sim[i])[::-1][:top_k]
        for rank, sid in enumerate(order, start=1):
            rows.append({
                "action_id": i + 1,
                "action_title": act_df.iloc[i]["action_title"],
                "matched_strategy_rank": rank,
                "strategy_title": strat_df.iloc[sid]["strategy_title"],
                "cosine_similarity": float(sim[i, sid]),
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

    return df, best, per_strategy, overall, emb_label

# -----------------------------
# GPT suggestion
# -----------------------------
def gpt_suggestion(strategy_title: str, action_title: str, similarity: float) -> str:
    if not client:
        return "OPENAI_API_KEY not set. Add it in Streamlit Cloud → App → Settings → Secrets."

    prompt = f"""
You are an expert strategy execution consultant.

Strategic Objective:
{strategy_title}

Current Action (low alignment; similarity={similarity:.3f}):
{action_title}

Return in this exact structure:
1) Improved action (1–2 sentences)
2) Suggested KPI(s) (3–5 bullet points)
3) Missing tasks / steps (3–7 bullet points)
4) Timeline adjustment (1 sentence)
5) Owner/role suggestion (1 line)
""".strip()

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
st.caption("Upload plans → similarity mapping → dashboard → live GPT improvements (cloud-safe).")

with st.sidebar:
    st.header("Upload")
    strat_file = st.file_uploader(
        "Strategic Plan (.md/.txt/.docx/.pdf)",
        type=["md","txt","docx","pdf"],
        key="strat_uploader"
    )
    act_file   = st.file_uploader(
        "Action Plan (.md/.txt/.docx/.pdf)",
        type=["md","txt","docx","pdf"],
        key="act_uploader"
    )

    st.header("Settings")
    top_k = st.slider("Top-K matches", 1, 5, 3, key="top_k")
    threshold = st.slider("Low alignment threshold", 0.20, 0.90, 0.60, 0.01, key="threshold")
    st.markdown("---")
    st.write("Live GPT:", "✅ Enabled" if client else "❌ Add OPENAI_API_KEY in Secrets")

# Store uploaded bytes ONCE (prevents “file consumed on rerun”)
if strat_file is not None:
    st.session_state.strat_bytes = strat_file.getvalue()
    st.session_state.strat_name = strat_file.name

if act_file is not None:
    st.session_state.act_bytes = act_file.getvalue()
    st.session_state.act_name = act_file.name

if st.session_state.strat_bytes is None or st.session_state.act_bytes is None:
    st.info("Upload both documents to start.")
    st.stop()

# Always rebuild parsed data from stored bytes (stable across reruns)
strategic_text = read_file_from_bytes(st.session_state.strat_name, st.session_state.strat_bytes)
action_text    = read_file_from_bytes(st.session_state.act_name, st.session_state.act_bytes)

strat_df = normalize_strategic_objectives(strategic_text, n=10)
act_df   = extract_actions(action_text)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Strategic Objectives (Normalized)")
    st.write("Units:", len(strat_df))
    st.dataframe(strat_df[["strategy_title"]], use_container_width=True)
with c2:
    st.subheader("Actions Detected")
    st.write("Actions:", len(act_df))
    st.dataframe(act_df[["action_title"]].head(30), use_container_width=True)

# Run sync button -> save into session_state
if st.button("Run Synchronization", type="primary", key="run_sync"):
    with st.spinner("Computing similarity..."):
        all_matches, best, per_strategy, overall, emb_label = build_alignment(strat_df, act_df, top_k=top_k)

    st.session_state.sync_done = True
    st.session_state.all_matches = all_matches
    st.session_state.best = best
    st.session_state.per_strategy = per_strategy
    st.session_state.overall = overall
    st.session_state.emb_label = emb_label
    st.session_state.gpt_text = ""  # reset previous GPT output

# Render results IF already synced (this is what stops “jumping back”)
if st.session_state.sync_done and st.session_state.best is not None:
    all_matches = st.session_state.all_matches
    best        = st.session_state.best
    per_strategy= st.session_state.per_strategy
    overall     = st.session_state.overall
    emb_label   = st.session_state.emb_label

    st.success(f"Embedding method: {emb_label}")

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

    st.download_button("Download alignment_results.csv (Top-K matches)",
                       data=all_matches.to_csv(index=False).encode("utf-8"),
                       file_name="alignment_results.csv", mime="text/csv", key="dl_all")
    st.download_button("Download best_matches.csv",
                       data=best.to_csv(index=False).encode("utf-8"),
                       file_name="best_matches.csv", mime="text/csv", key="dl_best")
    st.download_button("Download strategy_alignment_summary.csv",
                       data=per_strategy.to_csv(index=False).encode("utf-8"),
                       file_name="strategy_alignment_summary.csv", mime="text/csv", key="dl_strat")

    st.subheader("Live Intelligent Improvements (GPT)")
    low = best[best["cosine_similarity"] < threshold].sort_values("cosine_similarity").copy()
    st.write(f"Low-alignment actions (< {threshold:.2f}):", len(low))

    if len(low) > 0:
        pick = st.selectbox(
            "Select low-alignment action",
            options=low["action_id"].tolist(),
            key="low_pick",
            format_func=lambda aid: f"Action {aid}: {low[low['action_id']==aid]['action_title'].values[0][:90]}"
        )
        row = low[low["action_id"] == pick].iloc[0]

        if st.button("Generate GPT Suggestion", key="gen_gpt"):
            with st.spinner("Calling GPT..."):
                try:
                    st.session_state.gpt_text = gpt_suggestion(
                        row["strategy_title"], row["action_title"], row["cosine_similarity"]
                    )
                except Exception as e:
                    st.session_state.gpt_text = f"GPT call failed: {e}"

        st.text_area("Suggestion Output", st.session_state.gpt_text, height=320, key="gpt_out")
    else:
        st.success("No low-alignment actions under current threshold.")
else:
    st.info("Click **Run Synchronization** to compute alignment.")
