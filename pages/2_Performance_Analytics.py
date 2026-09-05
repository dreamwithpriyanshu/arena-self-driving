import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Performance Analytics", page_icon="📊", layout="wide")

st.title("Performance Analytics: DQN vs SARSA")
st.markdown("Detailed breakdown of agent performance metrics and accuracy comparisons.")

if "training_history" not in st.session_state or not st.session_state.training_history:
    st.info("No data available. Go to **Model Training** to generate training history.")
    st.stop()

df = pd.DataFrame(st.session_state.training_history)

if df.empty:
    st.warning("No data to display.")
else:
    # ── Global Comparison ──────────────────────────────────────────────────
    st.subheader("Accuracy & Survival Comparison")
    col_c1, col_c2 = st.columns(2)

    base = alt.Chart(df).encode(
        x=alt.X('episode_num:Q', title="Episode"),
        color=alt.Color('vehicle:N', scale=alt.Scale(
            domain=['R', 'S'], range=['#00E5FF', '#FF9100']
        ), legend=alt.Legend(title="Algorithm (R=DQN, S=SARSA)"))
    )

    with col_c1:
        reward_chart = base.mark_line(point=True).encode(
            y=alt.Y('total_reward:Q', title="Total Reward (Accuracy)")
        ).properties(height=300)
        st.altair_chart(reward_chart, use_container_width=True)

    with col_c2:
        steps_chart = base.mark_line(point=True).encode(
            y=alt.Y('steps:Q', title="Survival Steps")
        ).properties(height=300)
        st.altair_chart(steps_chart, use_container_width=True)

    st.markdown("---")

    # ── Algorithm Specific Metrics ──────────────────────────────────────────
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
                ).properties(height=200)
                st.altair_chart(loss_chart, use_container_width=True)
                
            # Epsilon decay
            if "epsilon" in df_r.columns:
                eps_chart = alt.Chart(df_r).mark_line(color="#00E5FF").encode(
                    x=alt.X('episode_num:Q', title="Episode"),
                    y=alt.Y('epsilon:Q', title="Exploration Rate (ε)")
                ).properties(height=200)
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
                ).properties(height=200)
                st.altair_chart(td_chart, use_container_width=True)
                
            # Epsilon decay
            if "epsilon" in df_s.columns:
                eps_chart = alt.Chart(df_s).mark_line(color="#FF9100").encode(
                    x=alt.X('episode_num:Q', title="Episode"),
                    y=alt.Y('epsilon:Q', title="Exploration Rate (ε)")
                ).properties(height=200)
                st.altair_chart(eps_chart, use_container_width=True)
        else:
            st.write("No SARSA data.")

    st.markdown("---")
    st.subheader("Raw History Table")
    st.dataframe(df, use_container_width=True)
