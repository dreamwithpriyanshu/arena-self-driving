import altair as alt
import pandas as pd
import streamlit as st

from src.data.dashboard import history_files, load_histories


st.set_page_config(page_title="Performance Analytics", page_icon="A", layout="wide")
st.title("Performance Analytics")
st.caption("Training history written by the headless trainer; no simulation is run in Streamlit.")

files = history_files()
if not files:
    st.info("No training history is available. Run the trainer first:")
    st.code(
        "python scripts/train.py --episodes 50 --warm-start",
        language="powershell",
    )
    st.stop()

selected = st.selectbox("History file", files, format_func=lambda path: path.name)
selected_paths = st.multiselect(
    "Compare runs", files, default=[selected], format_func=lambda path: path.name,
)
if not selected_paths:
    st.info("Select at least one timestamped run to analyse.")
    st.stop()
try:
    df = pd.DataFrame(load_histories(selected_paths))
except (OSError, ValueError) as exc:
    st.error(f"Could not load {selected.name}: {exc}")
    st.stop()

if df.empty:
    st.warning("The selected history file contains no records.")
    st.stop()

missing = {"episode_num", "vehicle"} - set(df.columns)
if missing:
    st.error(f"History is missing required columns: {', '.join(sorted(missing))}")
    st.stop()

df = df.sort_values(["run", "vehicle", "episode_num"])
st.write(f"Loaded `{selected.name}` · {len(df)} episode records")

COLORS = alt.Scale(domain=["SARSA"], range=["#00E5FF"])


def comparison_chart(field: str, title: str) -> alt.Chart | None:
    if field not in df.columns:
        return None
    return alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("episode_num:Q", title="Episode"),
        y=alt.Y(f"{field}:Q", title=title),
        color=alt.Color("vehicle:N", scale=COLORS, title="Vehicle"),
        strokeDash=alt.StrokeDash("run:N", title="Run"),
        tooltip=[
            "run:N",
            "vehicle:N",
            "episode_num:Q",
            alt.Tooltip(f"{field}:Q", title=title, format=".3f"),
        ],
    ).properties(height=280)


left, right = st.columns(2)
with left:
    chart = comparison_chart("total_reward", "Total reward")
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
with right:
    chart = comparison_chart("steps", "Survival steps")
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)

st.subheader("Learning signals")
for field, title in [
    ("avg_td_error", "SARSA average TD error"),
    ("epsilon", "Exploration rate"),
]:
    chart = comparison_chart(field, title)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)

st.subheader("Raw records")
st.dataframe(df, hide_index=True, use_container_width=True)
