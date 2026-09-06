import streamlit as st
from src.data.loader import get_dataset_summary
from pathlib import Path
import json

st.set_page_config(
    page_title="Arena Self-Driving — Data Center",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load training history (if available) into session state so the Streamlit
# multipage dashboard (Performance Analytics) can visualize recent training runs.
if "training_history" not in st.session_state:
    history_path = Path("artifacts") / "training_history.jsonl"
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8") as fh:
                lines = [line.strip() for line in fh if line.strip()]
                st.session_state.training_history = [json.loads(l) for l in lines]
        except Exception as e:
            st.session_state.training_history = []
            st.warning(f"Failed to load training history: {e}")
    else:
        st.session_state.training_history = []

# Provide a small control to refresh training history during a live session.
if st.sidebar.button("Reload Training History"):
    history_path = Path("artifacts") / "training_history.jsonl"
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8") as fh:
                lines = [line.strip() for line in fh if line.strip()]
                st.session_state.training_history = [json.loads(l) for l in lines]
            st.sidebar.success("Training history reloaded.")
        except Exception as e:
            st.sidebar.error(f"Failed to reload: {e}")
    else:
        st.sidebar.info("No training history file found yet.")

st.title("Data Command Center")

st.markdown("""
Welcome to the Arena Self-Driving dashboard. 

This interface is dedicated to **data viewing and performance analytics**. 
To ensure maximum performance and 60FPS physics, all simulation and gameplay has been moved to native desktop windows.
""")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Human Demonstrations Dataset")
    summary = get_dataset_summary()
    
    st.metric("Total Collected Episodes", summary["num_episodes"])
    st.metric("Total Transitions (Steps)", f"{summary['total_transitions']:,}")
    
    r_count = summary["vehicles"].get("R", 0)
    s_count = summary["vehicles"].get("S", 0)
    
    st.markdown(f"""
    **Breakdown by Agent:**
    * **R (DQN)**: {r_count} episodes
    * **S (SARSA)**: {s_count} episodes
    """)

with col2:
    st.subheader("🎮 How to Collect Data (Native Play)")
    st.markdown("""
    To record new driving demonstrations, use the native PyGame interface from your terminal. 
    This provides a lag-free 60FPS experience and automatically saves data to the dataset.

    **For DQN (R):**
    ```bash
    python scripts/play_human.py R
    ```

    **For SARSA (S):**
    ```bash
    python scripts/play_human.py S
    ```
    """)

    st.subheader("🤖 How to Watch AI (Native Play)")
    st.markdown("""
    To watch the trained agents drive autonomously:

    ```bash
    python scripts/play_agent.py R
    # or
    python scripts/play_agent.py S
    ```
    """)

# Global CSS for the UI Design System
st.markdown("""
<style>
    .stApp {
        background-color: #1E1E24;
        color: #E0E0E0;
    }
    [data-testid="stMetricValue"] {
        font-family: monospace;
        color: #00E5FF;
    }
</style>
""", unsafe_allow_html=True)
