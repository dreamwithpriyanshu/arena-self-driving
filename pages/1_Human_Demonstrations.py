import altair as alt
import pandas as pd
import streamlit as st

from src.data.loader import get_dataset_summary


st.set_page_config(page_title="Human Demonstrations", page_icon="R", layout="wide")
st.title("Human Demonstrations")
st.caption("Arrow-key driving demonstrations used to warm-start the SARSA policy.")

summary = get_dataset_summary()
episodes = pd.DataFrame(summary["episodes"])

if episodes.empty:
    st.info("No validated demonstrations are available yet.")
    st.code("python scripts/play_human.py", language="powershell")
    st.stop()

st.dataframe(
    {
        "Measure": ["Episodes", "Transitions", "Total reward", "SARSA-labelled demonstrations"],
        "Value": [
            summary["num_episodes"],
            f"{summary['total_transitions']:,}",
            f"{summary['total_reward']:.2f}",
            summary["vehicles"].get("S", 0),
        ],
    },
    hide_index=True,
    use_container_width=True,
)

if "vehicle" in episodes.columns:
    chart = alt.Chart(episodes).mark_bar().encode(
        x=alt.X("vehicle:N", title="Recorded vehicle", sort=["S", "R"]),
        y=alt.Y("count():Q", title="Episodes"),
        color=alt.Color(
            "vehicle:N",
            scale=alt.Scale(domain=["S", "R"], range=["#00E5FF", "#6B7280"]),
            legend=None,
        ),
        tooltip=["vehicle:N", alt.Tooltip("count():Q", title="Episodes")],
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)

st.subheader("Episode metadata")
columns = [
    column
    for column in ["episode_id", "vehicle", "num_transitions", "total_reward", "start_time", "end_time"]
    if column in episodes
]
st.dataframe(episodes[columns], hide_index=True, use_container_width=True)
