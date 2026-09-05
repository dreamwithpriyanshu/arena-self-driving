import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Documentation", page_icon="📚", layout="wide")

st.title("Project Documentation")
st.markdown("All project documentation is available here for easy review.")

docs_dir = Path("docs")

docs_files = [
    "project_overview.md",
    "architecture.md",
    "algorithm_notes.md",
    "ui_design_system.md",
    "security_notes.md",
    "training_manual.md",
    "human_training_7_day_plan.md"
]

tabs = st.tabs([f.replace(".md", "").replace("_", " ").title() for f in docs_files])

for tab, file_name in zip(tabs, docs_files):
    file_path = docs_dir / file_name
    with tab:
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            st.markdown(content)
        else:
            st.warning(f"Document `{file_name}` not found in `docs/` yet.")
