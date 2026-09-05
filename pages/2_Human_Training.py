import streamlit as st
from src.simulation import HumanFacade, render_highway_svg

st.set_page_config(page_title="Human Training", page_icon="🎮", layout="wide")

st.title("Human Training Recorder")
st.markdown("Record driving demonstrations to warm-start the agents.")

if "human_facade" not in st.session_state:
    st.session_state.human_facade = HumanFacade()

facade: HumanFacade = st.session_state.human_facade

col1, col2 = st.columns([1, 4])

with col1:
    st.subheader("Session Control")
    vehicle_choice = st.radio("Recording for:", ["R (DQN)", "S (SARSA)"])
    v_id = "R" if "R" in vehicle_choice else "S"
    
    if not facade.episode_active:
        if st.button("Start Recording", type="primary", use_container_width=True):
            facade.start(vehicle=v_id)
            st.rerun()
    else:
        st.warning("Recording Active 🔴")
        
        if st.button("Save Episode", type="primary", use_container_width=True):
            path = facade.save()
            st.success(f"Saved to {path}")
            # Rerun doesn't clear success immediately unless we delay
        
        if st.button("Discard Episode", use_container_width=True):
            facade.discard()
            st.error("Episode discarded.")
            
    st.markdown("---")
    st.subheader("Drive (Click Buttons)")
    # Since Streamlit cannot easily capture physical keydown events natively without custom HTML/JS hacks, 
    # we simulate the keyboard_controller via buttons.
    
    if facade.episode_active and not facade.episode_done:
        k_up = st.button("⬆️ Accelerate", use_container_width=True)
        k_down = st.button("⬇️ Brake", use_container_width=True)
        cols = st.columns(2)
        k_left = cols[0].button("⬅️ Left", use_container_width=True)
        k_right = cols[1].button("➡️ Right", use_container_width=True)
        k_space = st.button("⏸️ IDLE (Coast)", use_container_width=True)
        
        # Process action
        if k_up: facade.act("arrowup"); st.rerun()
        if k_down: facade.act("arrowdown"); st.rerun()
        if k_left: facade.act("arrowleft"); st.rerun()
        if k_right: facade.act("arrowright"); st.rerun()
        if k_space: facade.act("spacebar"); st.rerun()

with col2:
    st.subheader("Live View")
    if facade.episode_active:
        raw_state = facade.raw_state
        lane = facade.lane_index
        
        if raw_state is not None:
            svg_html = render_highway_svg(
                raw_state=raw_state.tolist(),
                ego_lane=lane,
                vehicle_type=v_id
            )
            import streamlit.components.v1 as components
            components.html(svg_html, height=320)
            
        st.metric("Steps Recorded", facade.step_count)
        
        if facade.episode_done:
            st.error("Episode ended (collision or time limit). Please Save or Discard.")
    else:
        st.info("Start a recording to see the view.")

# Inject JavaScript to map physical keyboard arrow keys to the Streamlit buttons
import streamlit.components.v1 as components
components.html("""
<script>
const doc = window.parent.document;
doc.addEventListener('keydown', function(e) {
    let targetText = null;
    switch(e.key) {
        case 'ArrowUp': targetText = 'Accelerate'; break;
        case 'ArrowDown': targetText = 'Brake'; break;
        case 'ArrowLeft': targetText = 'Left'; break;
        case 'ArrowRight': targetText = 'Right'; break;
        case ' ': targetText = 'IDLE'; break;
    }
    if (targetText) {
        // Prevent default scrolling for arrows and space
        e.preventDefault();
        const buttons = Array.from(doc.querySelectorAll('button'));
        const btn = buttons.find(b => b.innerText.includes(targetText));
        if (btn) btn.click();
    }
});
</script>
""", height=0, width=0)
