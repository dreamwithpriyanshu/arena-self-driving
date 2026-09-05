"""
Live Tracking — Watch agents drive autonomously in real-time.
"""
import time
import streamlit as st
import streamlit.components.v1 as components
from src.simulation import EnvManager, render_highway_svg
from src.simulation.training_facade import TrainingFacade

st.set_page_config(page_title="Live Tracking", page_icon="📡", layout="wide")

st.title("Live Tracking")
st.markdown("Watch the agents drive autonomously in real-time.")

if "training_facade" not in st.session_state:
    st.session_state.training_facade = TrainingFacade()

facade: TrainingFacade = st.session_state.training_facade

col1, col2 = st.columns([1, 4])

with col1:
    st.subheader("Controls")
    vehicle_choice = st.radio("Agent:", ["R (DQN)", "S (SARSA)"])
    v_id = "R" if "R" in vehicle_choice else "S"

    speed = st.slider("Playback Speed", min_value=1, max_value=10, value=2)
    sleep_time = 1.0 / speed

    run_btn = st.button("Run Episode", type="primary", use_container_width=True)
    eval_btn = st.button("Evaluate (Greedy)", use_container_width=True)

with col2:
    st.subheader("Live View")

    svg_container = st.empty()
    metrics_container = st.empty()

    if not run_btn and not eval_btn:
        empty_html = render_highway_svg(raw_state=[], ego_lane=0, vehicle_type=v_id)
        with svg_container:
            components.html(empty_html, height=420)
        st.info("Select an agent and click Run Episode.")

if run_btn or eval_btn:
    env_mgr = EnvManager(render_mode="rgb_array")

    def live_update(result, metrics):
        action_idx = result.action_taken if result.action_taken is not None else 1
        from src.envs.actions import ACTION_NAMES
        action_name = ACTION_NAMES.get(action_idx, "IDLE")

        svg_html = render_highway_svg(
            raw_state=result.raw_state.tolist(),
            ego_lane=result.lane_index,
            vehicle_type=v_id,
            speed=result.speed,
            last_action=action_name,
            reward=result.reward,
        )
        with svg_container:
            components.html(svg_html, height=420)

        with metrics_container.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Speed (m/s)", f"{result.speed:.1f}")
            m2.metric("Lane", result.lane_index)
            m3.metric("Reward", f"{result.reward:.2f}")
            if "loss" in metrics:
                m4.metric("Loss", f"{metrics['loss']:.4f}")
            elif "td_error" in metrics:
                m4.metric("TD Error", f"{metrics['td_error']:.4f}")
            else:
                m4.metric("Status", "Eval")

        time.sleep(sleep_time)

    with st.spinner("Episode Running..."):
        if run_btn:
            summary = facade.train_episode(
                vehicle=v_id,
                env_mgr=env_mgr,
                step_callback=live_update
            )
        else:
            summary = facade.evaluate_episode(
                vehicle=v_id,
                env_mgr=env_mgr,
                step_callback=live_update
            )

    env_mgr.close()

    if summary["terminated"]:
        st.error(f"Episode Terminated (Collision) after {summary['steps']} steps. Reward: {summary['total_reward']:.2f}")
    else:
        st.warning(f"Episode Truncated (Time Limit) after {summary['steps']} steps. Reward: {summary['total_reward']:.2f}")
