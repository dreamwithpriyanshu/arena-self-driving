# User Interface Design

The application uses a dark road-focused palette. SARSA (`S`) and active
controls use cyan `#00E5FF`; destructive stop actions use a contrasting red.

## Native windows

PyGame owns keyboard driving, simulation rendering, pause, road pace, HUD,
and save/discard interaction. Human demonstration and SARSA playback remain
native windows because browser controls cannot replace desktop game rendering.

## Browser workbench

FastAPI serves a responsive, no-build frontend with four sections:

| Section | Purpose |
|---|---|
| Drive | Launch human demonstration or SARSA playback and show demonstration totals. |
| Train | Configure the sole SARSA model, start or stop one run, and view state. |
| Analytics | Plot reward, survival steps, epsilon, and TD error for live or saved runs. |
| Documentation | Read the project Markdown notes without leaving the workbench. |

The frontend deliberately has no Node.js build step or third-party chart
runtime. Canvas charts keep the local control surface lightweight.
