from pathlib import Path

import streamlit as st

from src.data.dashboard import history_files, load_history, load_latest_history
from src.data.loader import get_dataset_summary


st.set_page_config(
    page_title="Arena Self-Driving — Overview",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


def configure_history_sidebar() -> None:
    """Expose the available training runs without starting any training."""
    if "training_history" not in st.session_state:
        records, selected_path = load_latest_history()
        st.session_state.training_history = records
        st.session_state._training_history_file = selected_path

    st.sidebar.header("Training evidence")
    files = history_files()
    if not files:
        st.sidebar.info("No training history runs found in artifacts/.")
        return

    selected = st.sidebar.selectbox("History file", files, format_func=lambda path: path.name)
    if st.sidebar.button("Load selected run"):
        try:
            st.session_state.training_history = load_history(selected)
            st.session_state._training_history_file = str(selected)
            st.sidebar.success("Loaded")
        except (OSError, ValueError) as exc:
            st.sidebar.error(f"Could not load run: {exc}")

    meta_path = selected.with_suffix(".meta.json")
    if meta_path.exists():
        st.sidebar.caption(f"Metadata: {meta_path.name}")


configure_history_sidebar()

st.title("Arena Self-Driving")
st.caption("Overview · shared human demonstrations · DQN versus SARSA")
st.markdown(
    """
This is the evidence desk for the experiment. It reads demonstrations and
training history from disk; driving, training, and agent playback happen in
native PyGame or headless CLI sessions.
"""
)

summary = get_dataset_summary()
st.subheader("Current evidence")
st.dataframe(
    {
        "Source": ["Human demonstrations", "Latest training history"],
        "Records": [summary["num_episodes"], len(st.session_state.training_history)],
        "Location": [
            "data/human_demonstrations",
            st.session_state.get("_training_history_file") or "No run loaded",
        ],
    },
    hide_index=True,
    use_container_width=True,
)

st.subheader("Native workflow")
left, right = st.columns(2)
with left:
    st.markdown("**Collect demonstrations**")
    st.code(
        "python scripts/play_human.py R\npython scripts/play_human.py S",
        language="powershell",
    )
    st.caption("Arrow keys drive the selected vehicle. ENTER starts; ESC discards the current run.")

with right:
    st.markdown("**Train and watch agents**")
    st.code(
        "python scripts/train.py --agent BOTH --episodes 50 --warm-start --history-mode per-run\n"
        "python scripts/play_multi_agent.py",
        language="powershell",
    )
    st.caption("Use the native windows for gameplay; return here to inspect the resulting files.")

st.markdown(
    """
<style>
    .stApp { background-color: #1E1E24; }
    [data-testid="stSidebar"] { background-color: #282830; }
    code, pre { font-family: "Cascadia Code", Consolas, monospace; }
    h1, h2, h3 { letter-spacing: -0.02em; }
</style>
""",
    unsafe_allow_html=True,
)
