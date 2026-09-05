import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Performance Analytics", page_icon="📊", layout="wide")

st.title("Performance Analytics")
st.markdown("Detailed breakdown of agent performance metrics.")

if "training_history" not in st.session_state or not st.session_state.training_history:
    st.info("No data available. Go to **Train & Compare** to generate training history.")
    st.stop()

df = pd.DataFrame(st.session_state.training_history)

col1, col2 = st.columns(2)

with col1:
    st.subheader("DQN (R) Metrics")
    df_r = df[df["vehicle"] == "R"]
    if not df_r.empty:
        # Loss chart
        if "avg_loss" in df_r.columns:
            loss_chart = alt.Chart(df_r).mark_line(color="#00E5FF").encode(
                x=alt.X('episode_num:Q', title="Episode"),
                y=alt.Y('avg_loss:Q', title="Average Loss")
            ).properties(height=250)
            st.altair_chart(loss_chart, use_container_width=True)
            
        # Epsilon decay
        if "epsilon" in df_r.columns:
            eps_chart = alt.Chart(df_r).mark_line(color="#00E5FF").encode(
                x=alt.X('episode_num:Q', title="Episode"),
                y=alt.Y('epsilon:Q', title="Exploration Rate (ε)")
            ).properties(height=250)
            st.altair_chart(eps_chart, use_container_width=True)
    else:
        st.write("No DQN data.")

with col2:
    st.subheader("SARSA (S) Metrics")
    df_s = df[df["vehicle"] == "S"]
    if not df_s.empty:
        # TD Error chart
        if "avg_td_error" in df_s.columns:
            td_chart = alt.Chart(df_s).mark_line(color="#FF9100").encode(
                x=alt.X('episode_num:Q', title="Episode"),
                y=alt.Y('avg_td_error:Q', title="Average TD Error")
            ).properties(height=250)
            st.altair_chart(td_chart, use_container_width=True)
            
        # Epsilon decay
        if "epsilon" in df_s.columns:
            eps_chart = alt.Chart(df_s).mark_line(color="#FF9100").encode(
                x=alt.X('episode_num:Q', title="Episode"),
                y=alt.Y('epsilon:Q', title="Exploration Rate (ε)")
            ).properties(height=250)
            st.altair_chart(eps_chart, use_container_width=True)
    else:
        st.write("No SARSA data.")

st.markdown("---")
st.subheader("Raw History Table")
st.dataframe(df, use_container_width=True)
