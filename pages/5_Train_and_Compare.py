import streamlit as st
import pandas as pd
import altair as alt
from src.simulation.training_facade import TrainingFacade
from src.simulation.env_manager import EnvManager

st.set_page_config(page_title="Train & Compare", page_icon="📈", layout="wide")

st.title("Train & Compare")
st.markdown("Run autonomous training batches and watch the metrics update live.")

if "training_facade" not in st.session_state:
    st.session_state.training_facade = TrainingFacade()
if "training_history" not in st.session_state:
    st.session_state.training_history = []

facade: TrainingFacade = st.session_state.training_facade

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Controls")
    
    if st.button("Warm-Start (Human Demos)", use_container_width=True):
        res = facade.warm_start()
        if res["loaded"] > 0:
            st.success(f"Loaded {res['loaded']} demonstrations.")
        else:
            st.warning("No valid demonstrations found.")
            
    st.markdown("---")
    
    num_episodes = st.number_input("Episodes per batch", min_value=1, max_value=50, value=5)
    
    train_r_btn = st.button("Train DQN (R)", type="primary", use_container_width=True)
    train_s_btn = st.button("Train SARSA (S)", type="primary", use_container_width=True)
    
    st.markdown("---")
    if st.button("Save Checkpoints", use_container_width=True):
        facade.save_checkpoints()
        st.success("Checkpoints saved.")
    
    if st.button("Clear History", use_container_width=True):
        st.session_state.training_history = []
        st.rerun()

with col2:
    st.subheader("Live Performance Curves")
    
    chart_container = st.empty()
    status_text = st.empty()
    
    def render_chart():
        if not st.session_state.training_history:
            chart_container.info("No training history yet. Run a batch!")
            return
            
        df = pd.DataFrame(st.session_state.training_history)
        
        # Base chart
        base = alt.Chart(df).encode(
            x=alt.X('episode_num:Q', title="Total Episodes"),
            color=alt.Color('vehicle:N', scale=alt.Scale(
                domain=['R', 'S'], range=['#00E5FF', '#FF9100']
            ))
        )
        
        # Reward chart
        reward_chart = base.mark_line(point=True).encode(
            y=alt.Y('total_reward:Q', title="Episode Reward")
        ).properties(height=200)
        
        # Steps chart
        steps_chart = base.mark_line(point=True).encode(
            y=alt.Y('steps:Q', title="Survival Steps")
        ).properties(height=200)
        
        # Combine vertically
        chart = alt.vconcat(reward_chart, steps_chart).resolve_scale(x='shared')
        
        chart_container.altair_chart(chart, use_container_width=True)

    # Initial render
    render_chart()
    
    if train_r_btn or train_s_btn:
        v_id = "R" if train_r_btn else "S"
        
        env_mgr = EnvManager(render_mode="rgb_array")
        
        # Determine starting episode number
        start_ep = len(st.session_state.training_history) + 1
        
        for i in range(num_episodes):
            status_text.text(f"Training {v_id} — Episode {i+1}/{num_episodes}...")
            
            # Run episode (we don't need a step callback for chart animation, 
            # we do incremental updates per episode)
            summary = facade.train_episode(vehicle=v_id, env_mgr=env_mgr)
            summary["episode_num"] = start_ep + i
            
            st.session_state.training_history.append(summary)
            render_chart()
            
        env_mgr.close()
        status_text.success(f"Completed {num_episodes} episodes for {v_id}.")
