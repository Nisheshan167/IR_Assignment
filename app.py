import os
import re
import io
import numpy as np
import pandas as pd
import streamlit as st

from docx import Document
from pypdf import PdfReader

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from openai import OpenAI

st.set_page_config(page_title="ISPS Dashboard", layout="wide")

# =========================================================
# OPENAI
# =========================================================
def get_openai_client():
    api_key = None

    # Streamlit Cloud secrets first
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None, "❌ No OPENAI_API_KEY found"

    try:
        client = OpenAI(api_key=api_key)
        return client, "✅ API key loaded"
    except Exception as e:
        return None, f"❌ Client init failed: {type(e).__name__}: {e}"

client, openai_status = get_openai_client()

# =========================================================
# SESSION STATE
# =========================================================
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

# =========================================================
# FILE READING
# =========================================================
def read_file_from_bytes(filename: str, data: bytes) -> str:
    name = filename.lower()

    if name.endswith(".txt") or name.endswith(".md"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs if p.text is not None])

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(pages)

    raise ValueError("Unsupported file type. Upload .txt/.md/.docx/.pdf")

# =========================================================
# PARSING
# =========================================================
def extract_actions(action_text: str) -> pd.DataFrame:
    actions = []

    # Pattern 1: markdown headings like ### Action ...
    parts = re.split(r"^###\s+", action_text, flags=re.MULTILINE)
    if len(parts) > 1:
        for p in parts[1:]:
            lines = [x.strip() for x in p.splitlines()]
            lines = [x for x in lines if x]
            title = lines[0] if lines else "Untitled Action"
            body = "\n".join(lines[1:]).strip()
            if title or body:
                actions.append({
                    "action_title": title,
                    "action_text": body if body else "No details provided."
                })

    # Pattern 2: Action N: ...
    if not actions:
        lines = action_text.splitlines()
        cur_title = None
        buf = []

        for line in lines:
            clean = line.strip()
            m = re.match(r"^(Action\s+\d+)\s*[:\-]\s*(.*)$", clean, flags=re.IGNORECASE)

            if m:
                if cur_title is not None:
                    actions.append({
                        "action_title": cur_title,
                        "action_text": "\n".join(buf).strip() if buf else "No details provided."
                    })
                cur_title = f"{m.group(1)}: {m.group(2)}".strip()
                buf = []
            else:
                if cur_title is not None:
                    buf.append(line)

        if cur_title is not None:
            actions.append({
                "action_title": cur_title,
                "action_text": "\n".join(buf).strip() if buf else "No details provided."
            })

    # Pattern 3: numbered sections as fallback
    if not actions:
        blocks = re.split(r"\n(?=\d+\.\s)", action_text)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            if len(title) > 3:
                actions.append({
                    "action_title": title,
                    "action_text": body if body else "No details provided."
                })

    df = pd.DataFrame(actions).head(300)

    if df.empty:
        df = pd.DataFrame([{
            "action_title": "No actions detected",
            "action_text": "The parser could not identify action items. Please check document formatting."
        }])

    return df

def normalize_strategic_objectives(strategic_text: str, n: int = 10) -> pd.DataFrame:
    # Prefer a strategic objectives section if found
    m = re.search(r"(Strategic Objectives.*)", strategic_text, flags=re.IGNORECASE | re.DOTALL)
    core = m.group(1) if m else strategic_text
    core = re.sub(r"\n{3,}", "\n\n", core).strip()

    # Try to split on headings first
    heading_chunks = re.split(
        r"(?=^(?:Strategy|Strategic Objective|Objective)\s+\d+[:\-\s])",
        core,
        flags=re.IGNORECASE | re.MULTILINE
    )
    heading_chunks = [c.strip() for c in heading_chunks if c.strip()]

    units = []

    if len(heading_chunks) >= 2:
        for idx, ch in enumerate(heading_chunks[:n], start=1):
            first_line = ch.splitlines()[0].strip()
            body = "\n".join(ch.splitlines()[1:]).strip()
            units.append({
                "strategy_title": first_line,
                "strategy_text": body if body else "No details provided."
            })
    else:
        # fallback chunking
        chunk_size = max(1500, len(core) // n) if n else max(1500, len(core))
        chunks = []
        i = 0
        while i < len(core) and len(chunks) < n:
            chunks.append(core[i:i + chunk_size].strip())
            i += chunk_size
        while len(chunks) < n:
            chunks.append("")

        for idx, ch in enumerate(chunks, start=1):
            units.append({
                "strategy_title": f"Strategic Objective {idx}",
                "strategy_text": ch if ch else "Details to be refined."
            })

    df = pd.DataFrame(units)

    if df.empty:
        df = pd.DataFrame([{
            "strategy_title": "Strategic Objective 1",
            "strategy_text": "No strategy content detected."
        }])

    return df

# =========================================================
# TF-IDF REPRESENTATION
# =========================================================
def embed_joint(strategy_texts, action_texts):
    joint = strategy_texts + action_texts

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    joint_emb = vectorizer.fit_transform(joint).toarray().astype(np.float32)

    norm = np.linalg.norm(joint_emb, axis=1, keepdims=True) + 1e-9
    joint_emb = joint_emb / norm

    strat_emb = joint_emb[:len(strategy_texts)]
    act_emb = joint_emb[len(strategy_texts):]

    return strat_emb, act_emb, "TF-IDF"

# =========================================================
# ALIGNMENT
# =========================================================
def build_alignment(strat_df: pd.DataFrame, act_df: pd.DataFrame, top_k: int = 3):
    strategy_texts = [
        f"{r.strategy_title}\n{r.strategy_text}"
        for r in strat_df.itertuples(index=False)
    ]
    action_texts = [
        f"{r.action_title}\n{r.action_text}"
        for r in act_df.itertuples(index=False)
    ]

    strat_emb, act_emb, emb_label = embed_joint(strategy_texts, action_texts)
    sim = cosine_similarity(act_emb, strat_emb)

    rows = []
    for i in range(sim.shape[0]):
        order = np.argsort(sim[i])[::-1][:top_k]
        for rank, sid in enumerate(order, start=1):
            rows.append({
                "action_id": i + 1,
                "action_title": act_df.iloc[i]["action_title"],
                "action_text": act_df.iloc[i]["action_text"],
                "matched_strategy_rank": rank,
                "strategy_title": strat_df.iloc[sid]["strategy_title"],
                "strategy_text": strat_df.iloc[sid]["strategy_text"],
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

# =========================================================
# GPT SUGGESTION
# =========================================================
def gpt_suggestion(
    strategy_title: str,
    strategy_text: str,
    action_title: str,
    action_text: str,
    similarity: float
) -> str:
    if client is None:
        return "OpenAI client not initialized. Add OPENAI_API_KEY in Streamlit secrets."

    prompt = f"""
You are an expert strategy execution consultant.

Strategic Objective Title:
{strategy_title}

Strategic Objective Details:
{strategy_text}

Current Action Title:
{action_title}

Current Action Details:
{action_text}

Current Similarity Score:
{similarity:.3f}

Please return in this exact structure:

1) Improved action
(Write 1-2 sentences)

2) Suggested KPI(s)
- bullet 1
- bullet 2
- bullet 3

3) Missing tasks / steps
- bullet 1
- bullet 2
- bullet 3

4) Timeline adjustment
(Write 1 sentence)

5) Owner/role suggestion
(Write 1 line)
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise business planning assistant. Be practical, concise, and structured."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=0.3,
        )

        if not resp.choices:
            return "OpenAI returned no choices."

        msg = resp.choices[0].message
        if not msg or not msg.content:
            return "OpenAI returned empty content."

        return msg.content.strip()

    except Exception as e:
        return f"OpenAI request failed: {type(e).__name__}: {e}"

# =========================================================
# UI
# =========================================================
st.title("ISPS — Intelligent Strategic Plan Synchronization System")
st.caption("Upload plans → similarity mapping → dashboard → live GPT improvements")

with st.sidebar:
    st.header("Upload")
    strat_file = st.file_uploader(
        "Strategic Plan (.md/.txt/.docx/.pdf)",
        type=["md", "txt", "docx", "pdf"],
        key="strat_uploader"
    )
    act_file = st.file_uploader(
        "Action Plan (.md/.txt/.docx/.pdf)",
        type=["md", "txt", "docx", "pdf"],
        key="act_uploader"
    )

    st.header("Settings")
    top_k = st.slider("Top-K matches", 1, 5, 3, key="top_k")
    threshold = st.slider("Low alignment threshold", 0.20, 0.90, 0.60, 0.01, key="threshold")

    st.markdown("---")
    st.write("Live GPT:", openai_status)
    st.write("Client ready:", client is not None)

# =========================================================
# STORE FILE BYTES ONCE
# =========================================================
if strat_file is not None:
    st.session_state.strat_bytes = strat_file.getvalue()
    st.session_state.strat_name = strat_file.name

if act_file is not None:
    st.session_state.act_bytes = act_file.getvalue()
    st.session_state.act_name = act_file.name

if st.session_state.strat_bytes is None or st.session_state.act_bytes is None:
    st.info("Upload both documents to start.")
    st.stop()

# =========================================================
# REBUILD FROM SESSION
# =========================================================
try:
    strategic_text = read_file_from_bytes(
        st.session_state.strat_name,
        st.session_state.strat_bytes
    )
    action_text = read_file_from_bytes(
        st.session_state.act_name,
        st.session_state.act_bytes
    )
except Exception as e:
    st.error(f"File reading failed: {type(e).__name__}: {e}")
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

if st.button("Run Synchronization", type="primary", key="run_sync"):
    with st.spinner("Computing similarity..."):
        try:
            all_matches, best, per_strategy, overall, emb_label = build_alignment(
                strat_df, act_df, top_k=top_k
            )

            st.session_state.sync_done = True
            st.session_state.all_matches = all_matches
            st.session_state.best = best
            st.session_state.per_strategy = per_strategy
            st.session_state.overall = overall
            st.session_state.emb_label = emb_label
            st.session_state.gpt_text = ""

        except Exception as e:
            st.error(f"Synchronization failed: {type(e).__name__}: {e}")

# =========================================================
# RESULTS
# =========================================================
if st.session_state.sync_done and st.session_state.best is not None:
    all_matches = st.session_state.all_matches
    best = st.session_state.best
    per_strategy = st.session_state.per_strategy
    overall = st.session_state.overall
    emb_label = st.session_state.emb_label

    st.success(f"Embedding method: {emb_label}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Overall Alignment (mean cosine)", f"{overall:.3f}")

    pct_good = float((best["cosine_similarity"] >= threshold).mean()) if len(best) else 0.0
    m2.metric(f"% Actions ≥ {threshold:.2f}", f"{pct_good * 100:.1f}%")
    m3.metric("Total Actions", f"{len(act_df)}")

    st.subheader("Strategy-wise Synchronization")
    st.dataframe(per_strategy, use_container_width=True)

    st.subheader("Best Match per Action")
    view = best.copy()
    view["band"] = pd.cut(
        view["cosine_similarity"],
        bins=[-1, threshold, 0.75, 1.01],
        labels=["Poor", "Medium", "Good"]
    )
    st.dataframe(
        view[["action_id", "action_title", "strategy_title", "cosine_similarity", "band"]],
        use_container_width=True
    )

    st.download_button(
        "Download alignment_results.csv (Top-K matches)",
        data=all_matches.to_csv(index=False).encode("utf-8"),
        file_name="alignment_results.csv",
        mime="text/csv",
        key="dl_all"
    )
    st.download_button(
        "Download best_matches.csv",
        data=best.to_csv(index=False).encode("utf-8"),
        file_name="best_matches.csv",
        mime="text/csv",
        key="dl_best"
    )
    st.download_button(
        "Download strategy_alignment_summary.csv",
        data=per_strategy.to_csv(index=False).encode("utf-8"),
        file_name="strategy_alignment_summary.csv",
        mime="text/csv",
        key="dl_strat"
    )

    st.subheader("Live Intelligent Improvements (GPT)")

    low = best[best["cosine_similarity"] < threshold].sort_values("cosine_similarity").copy()
    st.write(f"Low-alignment actions (< {threshold:.2f}):", len(low))

    if len(low) > 0:
        pick = st.selectbox(
            "Select low-alignment action",
            options=low["action_id"].tolist(),
            key="low_pick",
            format_func=lambda aid: (
                f"Action {aid}: "
                f"{low.loc[low['action_id'] == aid, 'action_title'].values[0][:90]}"
            )
        )

        row = low[low["action_id"] == pick].iloc[0]

        with st.expander("Selected low-alignment item", expanded=True):
            st.markdown(f"**Action:** {row['action_title']}")
            st.markdown(f"**Matched Strategy:** {row['strategy_title']}")
            st.markdown(f"**Similarity:** {row['cosine_similarity']:.3f}")

        if st.button("Generate GPT Suggestion", key="gen_gpt"):
            with st.spinner("Calling GPT..."):
                st.session_state.gpt_text = gpt_suggestion(
                    strategy_title=row["strategy_title"],
                    strategy_text=row["strategy_text"],
                    action_title=row["action_title"],
                    action_text=row["action_text"],
                    similarity=row["cosine_similarity"],
                )

        st.markdown("### Suggestion Output")
        st.text_area(
            "Generated recommendation",
            value=st.session_state.gpt_text,
            height=320,
            disabled=True
        )
    else:
        st.success("No low-alignment actions under current threshold.")
else:
    st.info("Click **Run Synchronization** to compute alignment.")
