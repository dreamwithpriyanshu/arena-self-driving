"""
Human Training Page — Record driving demonstrations for R (DQN) and S (SARSA).

Uses a proper browser keyboard component for real-time control,
an explicit state machine for episode lifecycle, and an in-road HUD.
"""
import streamlit as st
import streamlit.components.v1 as components
from src.simulation import HumanFacade, render_highway_svg
from src.data.loader import get_dataset_summary
from src.envs.actions import ACTION_NAMES

st.set_page_config(page_title="Human Training", page_icon="🎮", layout="wide")

# ── Session state init ──────────────────────────────────────────
if "human_facade" not in st.session_state:
    st.session_state.human_facade = HumanFacade()
if "ht_status" not in st.session_state:
    st.session_state.ht_status = "READY"  # READY | RECORDING | EPISODE_ENDED | SAVED | DISCARDED
if "ht_last_action" not in st.session_state:
    st.session_state.ht_last_action = ""
if "ht_last_reward" not in st.session_state:
    st.session_state.ht_last_reward = 0.0
if "ht_end_reason" not in st.session_state:
    st.session_state.ht_end_reason = ""
if "ht_end_steps" not in st.session_state:
    st.session_state.ht_end_steps = 0
if "ht_end_reward" not in st.session_state:
    st.session_state.ht_end_reward = 0.0

facade: HumanFacade = st.session_state.human_facade

# ── Header ──────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
    <h1 style="margin:0;padding:0;">Human Training</h1>
    <span style="background:#282830;border:1px solid #444;border-radius:4px;padding:2px 10px;font-family:Consolas,monospace;font-size:13px;color:#888;">
        SIMULATION LAB
    </span>
</div>
""", unsafe_allow_html=True)

# ── Layout: Sidebar | Main ─────────────────────────────────────
sidebar, main_area = st.columns([1, 4])

with sidebar:
    # ── Vehicle Selection ────────────────────────────────────
    st.markdown("#### Session")
    vehicle_choice = st.radio(
        "Human drives:",
        ["R — DQN", "S — SARSA"],
        index=0,
        disabled=(st.session_state.ht_status == "RECORDING"),
    )
    v_id = "R" if "R" in vehicle_choice else "S"
    algo = "DQN" if v_id == "R" else "SARSA"
    v_color = "#00E5FF" if v_id == "R" else "#FF9100"

    st.markdown(f"""
    <div style="background:#1a1a22;border:1px solid {v_color};border-radius:6px;padding:8px 12px;margin:4px 0 12px 0;font-family:Consolas,monospace;">
        <span style="color:{v_color};font-weight:bold;">HUMAN → {v_id}</span><br/>
        <span style="color:#888;font-size:12px;">{algo}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── State Machine Buttons ────────────────────────────────
    status = st.session_state.ht_status

    if status == "READY" or status == "SAVED" or status == "DISCARDED":
        if st.button("▶ START TRAINING", type="primary", use_container_width=True):
            facade.start(vehicle=v_id)
            st.session_state.ht_status = "RECORDING"
            st.session_state.ht_last_action = ""
            st.session_state.ht_last_reward = 0.0
            st.rerun()

    elif status == "RECORDING":
        st.markdown(f"""
        <div style="background:rgba(255,50,50,0.15);border:1px solid #FF1744;border-radius:6px;padding:6px 10px;text-align:center;font-family:Consolas,monospace;">
            <span style="color:#FF1744;font-weight:bold;">● RECORDING</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Steps: {facade.step_count} | Reward: {facade.total_reward:.2f}")

    elif status == "EPISODE_ENDED":
        st.markdown(f"""
        <div style="background:rgba(255,160,0,0.15);border:1px solid #FF9100;border-radius:6px;padding:6px 10px;text-align:center;font-family:Consolas,monospace;">
            <span style="color:#FF9100;">EPISODE ENDED</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Controls Reference ───────────────────────────────────
    st.markdown("#### Controls")
    controls_md = """
| Key | Action |
|-----|--------|
| `W` / `↑` | Accelerate |
| `S` / `↓` | Brake |
| `A` / `←` | Lane Left |
| `D` / `→` | Lane Right |
| `Space` | Idle |
| `R` | Reset Episode |
"""
    st.markdown(controls_md)

    st.markdown("---")

    # ── Training Data Summary ────────────────────────────────
    st.markdown("#### Training Data")
    summary = get_dataset_summary()
    st.metric("Saved Episodes", summary["num_episodes"])
    st.metric("Total Transitions", f"{summary['total_transitions']:,}")

    r_count = summary["vehicles"].get("R", 0)
    s_count = summary["vehicles"].get("S", 0)
    st.caption(f"R demos: {r_count} | S demos: {s_count}")

    if summary["total_transitions"] > 0:
        if st.button("🧠 Train from Human Data", use_container_width=True):
            with st.spinner("Warm-starting agents..."):
                from src.simulation.training_facade import TrainingFacade
                if "training_facade" not in st.session_state:
                    st.session_state.training_facade = TrainingFacade()
                tf = st.session_state.training_facade
                res = tf.warm_start()
            st.success(f"Trained! {res['loaded']} transitions loaded.")


# ── Main Area ───────────────────────────────────────────────────
with main_area:
    status = st.session_state.ht_status

    # ── Keyboard Component ───────────────────────────────────
    # Only active during RECORDING
    if status == "RECORDING":
        # Bi-directional keyboard component
        components.html("""
        <div id="kb-root" style="
            padding: 6px 14px; border-radius: 6px;
            background: #1a1a22; border: 1px solid #333;
            font-family: Consolas, monospace; font-size: 13px;
            color: #aaa; display: flex; align-items: center; gap: 10px;
            outline: none; cursor: pointer; user-select: none;
        " tabindex="0">
            <span id="kb-dot" style="width:10px;height:10px;border-radius:50%;background:#555;display:inline-block;"></span>
            <span id="kb-label">Click here, then use WASD / Arrow keys to drive</span>
            <span id="kb-action" style="margin-left:auto;color:#00E5FF;font-weight:bold;"></span>
        </div>
        <script>
        (function() {
            const root = document.getElementById('kb-root');
            const dot = document.getElementById('kb-dot');
            const label = document.getElementById('kb-label');
            const actionLabel = document.getElementById('kb-action');
            const held = new Set();
            let focused = false;

            const KEY_MAP = {
                'ArrowUp':'arrowup','ArrowDown':'arrowdown',
                'ArrowLeft':'arrowleft','ArrowRight':'arrowright',
                'w':'w','W':'w','a':'a','A':'a','s':'s_key','S':'s_key','d':'d','D':'d',
                ' ':' ','r':'r','R':'r'
            };
            const NAMES = {
                'arrowup':'ACCELERATE','arrowdown':'BRAKE',
                'arrowleft':'LEFT','arrowright':'RIGHT',
                'w':'ACCELERATE','s_key':'BRAKE','a':'LEFT','d':'RIGHT',
                ' ':'IDLE','r':'RESET'
            };
            const PREVENT = new Set(Object.keys(KEY_MAP));

            function setFocus(v) {
                focused = v;
                dot.style.background = v ? '#00E5FF' : '#555';
                label.textContent = v ? 'KEYBOARD ACTIVE' : 'Click here to activate keyboard';
                label.style.color = v ? '#00E5FF' : '#aaa';
                root.style.borderColor = v ? '#00E5FF' : '#333';
            }
            root.addEventListener('focus', () => setFocus(true));
            root.addEventListener('blur', () => { setFocus(false); held.clear(); actionLabel.textContent=''; });
            root.addEventListener('click', () => root.focus());
            setTimeout(() => root.focus(), 200);

            root.addEventListener('keydown', function(e) {
                if (!focused) return;
                if (PREVENT.has(e.key)) e.preventDefault();
                let mapped = KEY_MAP[e.key];
                if (!mapped) return;
                if (held.has(mapped)) return;
                held.add(mapped);
                actionLabel.textContent = NAMES[mapped] || mapped;

                // Map s_key back to s for Python
                let pyKey = mapped === 's_key' ? 's' : mapped;

                // Find all Streamlit buttons and click the hidden action receiver
                const parentDoc = window.parent.document;
                const buttons = parentDoc.querySelectorAll('button');
                for (const btn of buttons) {
                    if (btn.textContent.trim() === '__ACTION__' + pyKey) {
                        btn.click();
                        break;
                    }
                }
            });
            root.addEventListener('keyup', function(e) {
                let mapped = KEY_MAP[e.key];
                if (mapped) held.delete(mapped);
                if (held.size === 0) actionLabel.textContent = '';
            });
        })();
        </script>
        """, height=42)

        # Hidden action buttons — one per possible key.
        # The JS clicks these; Streamlit processes the rerun.
        action_keys = ["arrowup", "arrowdown", "arrowleft", "arrowright",
                        "w", "s", "a", "d", " ", "r"]
        # Use a container that won't be visible
        action_container = st.container()
        with action_container:
            cols = st.columns(len(action_keys))
            triggered_action = None
            for idx, key in enumerate(action_keys):
                with cols[idx]:
                    if st.button(f"__ACTION__{key}", key=f"act_{key}",
                                 label_visibility="collapsed"):
                        triggered_action = key

        # Process action
        if triggered_action is not None:
            if triggered_action == "r":
                # Reset: discard current episode, start fresh
                try:
                    facade.discard()
                except Exception:
                    pass
                facade.start(vehicle=v_id)
                st.session_state.ht_last_action = "RESET"
                st.session_state.ht_last_reward = 0.0
                st.rerun()
            else:
                try:
                    result = facade.act(triggered_action)
                    action_idx = result.action_taken if result.action_taken is not None else 1
                    st.session_state.ht_last_action = ACTION_NAMES.get(action_idx, "IDLE")
                    st.session_state.ht_last_reward = result.reward

                    if result.terminated or result.truncated:
                        reason = "COLLISION" if result.terminated else "TIMEOUT"
                        st.session_state.ht_status = "EPISODE_ENDED"
                        st.session_state.ht_end_reason = reason
                        st.session_state.ht_end_steps = facade.step_count
                        st.session_state.ht_end_reward = facade.total_reward
                except RuntimeError:
                    # Episode already done
                    st.session_state.ht_status = "EPISODE_ENDED"

                st.rerun()

    # ── Highway Visualization ────────────────────────────────
    if status == "RECORDING" or status == "EPISODE_ENDED":
        raw_state = facade.raw_state
        lane = facade.lane_index

        if raw_state is not None:
            speed_val = 0.0
            if facade.manager._prev_result:
                speed_val = facade.manager._prev_result.speed

            svg_html = render_highway_svg(
                raw_state=raw_state.tolist(),
                ego_lane=lane,
                vehicle_type=v_id,
                speed=speed_val,
                last_action=st.session_state.ht_last_action,
                step_count=facade.step_count,
                reward=st.session_state.ht_last_reward,
            )
            components.html(svg_html, height=420)

        # ── Live Telemetry (below road) ──────────────────────
        if status == "RECORDING":
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Steps", facade.step_count)
            t2.metric("Total Reward", f"{facade.total_reward:.2f}")
            t3.metric("Last Action", st.session_state.ht_last_action or "—")
            t4.metric("Last Reward", f"{st.session_state.ht_last_reward:+.2f}")

    # ── Episode Ended Panel ──────────────────────────────────
    if status == "EPISODE_ENDED":
        st.markdown("---")
        reason = st.session_state.ht_end_reason
        reason_color = "#FF1744" if reason == "COLLISION" else "#FF9100"

        st.markdown(f"""
        <div style="
            background: #1a1a22; border: 1px solid {reason_color}; border-radius: 8px;
            padding: 20px; font-family: Consolas, monospace; max-width: 500px;
        ">
            <div style="font-size:18px;font-weight:bold;color:{reason_color};margin-bottom:12px;">
                EPISODE COMPLETE
            </div>
            <div style="color:#888;margin-bottom:12px;">
                Reason: <span style="color:{reason_color};font-weight:bold;">{reason}</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;color:#ccc;">
                <div>Steps: <strong>{st.session_state.ht_end_steps}</strong></div>
                <div>Reward: <strong>{st.session_state.ht_end_reward:.2f}</strong></div>
                <div>Vehicle: <strong>{v_id}</strong></div>
                <div>Algorithm: <strong>{algo}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("💾 SAVE EPISODE", type="primary", use_container_width=True):
                try:
                    path = facade.save()
                    st.session_state.ht_status = "SAVED"
                    st.rerun()
                except Exception as e:
                    st.error(f"Save failed: {e}")
        with c2:
            if st.button("🗑️ DISCARD", use_container_width=True):
                try:
                    facade.discard()
                except Exception:
                    pass
                st.session_state.ht_status = "DISCARDED"
                st.rerun()
        with c3:
            if st.button("🔄 NEW EPISODE", use_container_width=True):
                try:
                    facade.discard()
                except Exception:
                    pass
                facade.start(vehicle=v_id)
                st.session_state.ht_status = "RECORDING"
                st.session_state.ht_last_action = ""
                st.session_state.ht_last_reward = 0.0
                st.rerun()

    # ── Ready / Saved / Discarded State ──────────────────────
    if status == "READY":
        st.markdown("""
        <div style="
            text-align:center; padding:80px 20px;
            color:#555; font-family: Consolas, monospace;
        ">
            <div style="font-size:48px;margin-bottom:16px;">🚗</div>
            <div style="font-size:16px;">Select a vehicle and click START TRAINING</div>
            <div style="font-size:13px;margin-top:8px;color:#444;">
                Use WASD or Arrow keys to drive after starting
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif status == "SAVED":
        st.success("Episode saved successfully! Click START TRAINING to record another.")

    elif status == "DISCARDED":
        st.warning("Episode discarded. Click START TRAINING to try again.")
