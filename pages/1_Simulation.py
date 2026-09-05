import streamlit as st
import numpy as np
from src.simulation import EnvManager, render_highway_svg

st.set_page_config(page_title="Simulation", page_icon="🚗", layout="wide")

st.title("Simulation Inspector")
st.markdown("Manually step through the highway environment to verify state dynamics.")

# Initialize in session state so it persists across button clicks
if "env_mgr" not in st.session_state:
    st.session_state.env_mgr = EnvManager(render_mode="rgb_array")
    st.session_state.current_state = st.session_state.env_mgr.reset(seed=42)

env_mgr: EnvManager = st.session_state.env_mgr
current_state = st.session_state.current_state

col1, col2 = st.columns([1, 4])

with col1:
    st.subheader("Controls")
    if st.button("Reset Environment", use_container_width=True):
        st.session_state.current_state = env_mgr.reset()
        st.rerun()
        
    st.markdown("---")
    st.markdown("**Actions**")
    
    # 0=L, 1=IDLE, 2=R, 3=FAST, 4=SLOW
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Left (0)", use_container_width=True):
            st.session_state.current_state = env_mgr.step(0)
            st.rerun()
        if st.button("IDLE (1)", use_container_width=True):
            st.session_state.current_state = env_mgr.step(1)
            st.rerun()
        if st.button("Slow (4)", use_container_width=True):
            st.session_state.current_state = env_mgr.step(4)
            st.rerun()
    with action_col2:
        if st.button("Right (2)", use_container_width=True):
            st.session_state.current_state = env_mgr.step(2)
            st.rerun()
        if st.button("Fast (3)", use_container_width=True):
            st.session_state.current_state = env_mgr.step(3)
            st.rerun()

with col2:
    st.subheader("Live View")
    
    # Render SVG
    if current_state:
        svg_html = render_highway_svg(
            raw_state=current_state.raw_state.tolist(),
            ego_lane=current_state.lane_index,
            vehicle_type="R" # Default cyan for inspector
        )
        st.markdown(svg_html, unsafe_allow_html=True)
        
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
