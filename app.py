import streamlit as st

st.set_page_config(
    page_title="Self-Driving Car Simulation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Self-Driving Car MVP")

st.markdown("""
Welcome to the Self-Driving Car Simulation MVP.

This dashboard allows you to:
- **1. Simulation**: View the raw environment and manually step through it.
- **2. Human Training**: Record demonstrations using on-screen controls to train the agents.
- **3. Live Tracking**: Watch the autonomous agents (DQN and SARSA) drive in real-time.
- **4. Performance Analytics**: Compare the training metrics (Reward, Loss, TD Error) of both models.
- **5. Train & Compare**: Warm-start from human data and run autonomous training loops.
- **6. Docs**: Read the architecture, security, and algorithm documentation.

Use the sidebar to navigate between pages.
""")

# Global CSS for the UI Design System (Dark asphalt)
st.markdown("""
<style>
    /* Force dark theme elements */
    .stApp {
        background-color: #1E1E24;
        color: #E0E0E0;
    }
    
    /* Monospace metrics */
    [data-testid="stMetricValue"] {
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)
