"""
Simulation Inspector — manually step through the highway environment.
"""
import streamlit as st
import streamlit.components.v1 as components
from src.simulation import EnvManager, render_highway_svg
from src.envs.actions import ACTION_NAMES

st.set_page_config(page_title="Simulation", page_icon="🚗", layout="wide")

st.title("Simulation Inspector")
st.markdown("Manually step through the highway environment to verify state dynamics.")

if "env_mgr" not in st.session_state:
    st.session_state.env_mgr = EnvManager(render_mode="rgb_array")
    st.session_state.current_state = st.session_state.env_mgr.reset(seed=42)
if "sim_last_action" not in st.session_state:
    st.session_state.sim_last_action = ""

env_mgr: EnvManager = st.session_state.env_mgr
current_state = st.session_state.current_state

col1, col2 = st.columns([1, 4])

with col1:
    st.subheader("Controls")
    if st.button("Reset Environment", use_container_width=True):
        st.session_state.current_state = env_mgr.reset()
        st.session_state.sim_last_action = ""
        st.rerun()

    st.markdown("---")
    st.markdown("**Actions**")

    # 0=L, 1=IDLE, 2=R, 3=FAST, 4=SLOW
    if st.button("⬆ Accelerate", use_container_width=True):
        st.session_state.current_state = env_mgr.step(3)
        st.session_state.sim_last_action = "FASTER"
        st.rerun()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅ Left", use_container_width=True):
            st.session_state.current_state = env_mgr.step(0)
            st.session_state.sim_last_action = "LANE_LEFT"
            st.rerun()
    with c2:
        if st.button("Right ➡", use_container_width=True):
            st.session_state.current_state = env_mgr.step(2)
            st.session_state.sim_last_action = "LANE_RIGHT"
            st.rerun()
    if st.button("⬇ Brake", use_container_width=True):
        st.session_state.current_state = env_mgr.step(4)
        st.session_state.sim_last_action = "SLOWER"
        st.rerun()
    if st.button("⏸ IDLE", use_container_width=True):
        st.session_state.current_state = env_mgr.step(1)
        st.session_state.sim_last_action = "IDLE"
        st.rerun()

with col2:
    st.subheader("Live View")

    if current_state:
        svg_html = render_highway_svg(
            raw_state=current_state.raw_state.tolist(),
            ego_lane=current_state.lane_index,
            vehicle_type="R",
            speed=current_state.speed,
            last_action=st.session_state.sim_last_action,
            step_count=env_mgr.step_count,
            reward=current_state.reward,
        )
        components.html(svg_html, height=420)

    # Telemetry
    if current_state:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Step", env_mgr.step_count)
        m2.metric("Speed (m/s)", f"{current_state.speed:.1f}")
        m3.metric("Lane", current_state.lane_index)
        m4.metric("Reward", f"{current_state.reward:.2f}")

        if current_state.terminated:
            st.error("Collision! Episode Terminated.")
        elif current_state.truncated:
            st.warning("Time limit reached! Episode Truncated.")
