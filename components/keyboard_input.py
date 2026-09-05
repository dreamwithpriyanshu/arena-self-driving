"""
Browser keyboard input component for Streamlit.

Returns the latest pressed action key to Python without requiring
button clicks or full page reruns. Uses streamlit.components.v1.html
with bi-directional communication via Streamlit.setComponentValue().
"""

import streamlit.components.v1 as components

KEYBOARD_COMPONENT_HTML = """
<div id="kb-root" style="
    padding: 8px 14px;
    border-radius: 6px;
    background: #1a1a22;
    border: 1px solid #333;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    color: #aaa;
    display: flex;
    align-items: center;
    gap: 10px;
    outline: none;
    cursor: pointer;
    user-select: none;
" tabindex="0">
    <span id="kb-dot" style="
        width: 10px; height: 10px; border-radius: 50%;
        background: #555; display: inline-block;
    "></span>
    <span id="kb-label">Click here to activate keyboard</span>
    <span id="kb-action" style="
        margin-left: auto; color: #00E5FF; font-weight: bold;
    "></span>
</div>

<script>
(function() {
    const root = document.getElementById('kb-root');
    const dot = document.getElementById('kb-dot');
    const label = document.getElementById('kb-label');
    const actionLabel = document.getElementById('kb-action');

    // Track which keys are currently held to suppress repeats
    const held = new Set();
    let focused = false;
    let lastSent = null;

    const KEY_MAP = {
        'ArrowUp': 'arrowup', 'ArrowDown': 'arrowdown',
        'ArrowLeft': 'arrowleft', 'ArrowRight': 'arrowright',
        'w': 'w', 'W': 'w', 'a': 'a', 'A': 'a',
        's': 's', 'S': 's', 'd': 'd', 'D': 'd',
        ' ': ' ', 'r': 'r', 'R': 'r',
    };

    const ACTION_NAMES = {
        'arrowup': 'ACCELERATE', 'arrowdown': 'BRAKE',
        'arrowleft': 'LEFT', 'arrowright': 'RIGHT',
        'w': 'ACCELERATE', 's': 'BRAKE',
        'a': 'LEFT', 'd': 'RIGHT',
        ' ': 'IDLE', 'r': 'RESET',
    };

    const PREVENT = new Set([
        'ArrowUp','ArrowDown','ArrowLeft','ArrowRight',
        ' ','w','a','s','d','W','A','S','D','r','R'
    ]);

    function setFocused(val) {
        focused = val;
        dot.style.background = val ? '#00E5FF' : '#555';
        label.textContent = val ? 'KEYBOARD ACTIVE' : 'Click here to activate keyboard';
        label.style.color = val ? '#00E5FF' : '#aaa';
        root.style.borderColor = val ? '#00E5FF' : '#333';
    }

    root.addEventListener('focus', () => setFocused(true));
    root.addEventListener('blur', () => { setFocused(false); held.clear(); });
    root.addEventListener('click', () => root.focus());

    // Auto-focus on load
    setTimeout(() => root.focus(), 300);

    root.addEventListener('keydown', function(e) {
        if (!focused) return;
        if (PREVENT.has(e.key)) e.preventDefault();

        const mapped = KEY_MAP[e.key];
        if (!mapped) return;

        // Suppress key repeat
        if (held.has(mapped)) return;
        held.add(mapped);

        actionLabel.textContent = ACTION_NAMES[mapped] || mapped;

        // Send to Streamlit
        if (window.parent && window.parent.postMessage) {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: mapped
            }, '*');
        }
    });

    root.addEventListener('keyup', function(e) {
        const mapped = KEY_MAP[e.key];
        if (mapped) held.delete(mapped);
        if (held.size === 0) {
            actionLabel.textContent = '';
        }
    });
})();
</script>
"""


def render_keyboard_input(key: str = "kb_input", height: int = 45):
    """
    Render the keyboard input component.
    Returns the latest key action string, or None if no key pressed.
    """
    value = components.html(
        KEYBOARD_COMPONENT_HTML,
        height=height,
    )
    return value
