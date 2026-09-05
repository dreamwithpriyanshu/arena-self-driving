# Self-Driving Car Simulation MVP — Antigravity Build Prompt (v2)

## Why this revision exists

The original plan (R = DQN vs S = SARSA, trained on shared human demonstrations
in HighwayEnv, with a 7-day human-training phase after 5 build steps) is solid.
This version tightens five things that tend to go wrong on agent-built MVPs:
documentation drifting out of sync with code, architecture eroding into a pile
of scripts, quiet security/robustness bugs, "AI-generated dashboard" UI, and a
UI that has no real place to *watch* the cars or *see* the numbers move.

Keep everything from the original plan that isn't contradicted below. This
document is additive: it sharpens the rules, adds two new pages, and appends a
documentation + architecture + security checklist to every build step.

---

## 1. Guiding principles (read first, apply every step)

1. **Documentation is a deliverable of every phase, not a Day-5 afterthought.**
   Each build step ends only when its corresponding doc file is written or
   updated — not just when the code runs.
2. **Architecture over convenience.** Every phase must respect module
   boundaries (env / agents / data / training / evaluation / UI). No phase is
   allowed to reach across a boundary "just this once."
3. **Simple and safe beats clever.** A college-explainable function that
   validates its inputs beats a dense one-liner that doesn't.
4. **The UI must look like a simulation lab tool, not a SaaS landing page.**
   No unexplained gradients, no stock photography, no "AI dashboard" clichés.
   Where imagery is genuinely useful (icons, car sprites, a track texture,
   a favicon), generate small, consistent, purpose-built assets instead of
   pulling stock images.
5. **The comparison must be watchable, not just loggable.** A reviewer should
   be able to open the app and *see* the cars driving live, and *see* R vs S
   performance without reading a JSON file.

---

## 2. Updated non-negotiable rules (adds to the original list)

- Documentation: every module gets a one-paragraph docstring-level summary in
  `docs/`, not just inline code. No module ships undocumented.
- Architecture: no file may import directly across more than one layer
  (e.g., a Streamlit page must not reach into `src/envs/` internals directly —
  it goes through a defined interface in `src/simulation/`).
- Security & robustness:
  - No `eval`/`exec`/`pickle.load` on files that weren't written by this
    app's own trusted save functions.
  - All file paths built from user input (episode names, uploaded files) must
    be sanitized — no path traversal, no arbitrary file overwrite.
  - All Streamlit inputs (numbers, file uploads, dropdowns) are validated and
    bounded before being used in simulation or training.
  - No secrets, API keys, or credentials in the repo, ever — this project
    needs none, so if code ever calls out to a network service, treat that as
    a design mistake, not a config detail to hide.
  - Every environment/session resource (`env.close()`, file handles, threads)
    must be released, including on exceptions — use context managers.
  - Failures must be visible: log and surface errors in the UI; never
    silently swallow an exception with a bare `except: pass`.
- UI/assets: any generated image asset (icon, sprite, texture) is produced
  once, saved under `assets/`, and reused — never generated at request time
  inside the simulation loop.
- Animation: motion in the UI must be lightweight (CSS transitions, SVG/canvas
  redraws, or Plotly/Altair animated frames) — no heavy 3D engines, no
  frame-by-frame video rendering for the MVP.

---

## 3. Updated project structure

```
self_driving_car_mvp/
├── app.py
├── pages/
│   ├── 1_Simulation.py
│   ├── 2_Human_Training.py
│   ├── 3_Live_Tracking.py
│   ├── 4_Performance_Analytics.py
│   ├── 5_Train_and_Compare.py
│   └── 6_Docs.py
├── src/
│   ├── envs/
│   ├── agents/
│   ├── human/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── simulation/        # the ONLY layer pages are allowed to call into
├── components/
├── configs/
├── assets/                 # generated icons/sprites/textures, checked in once
├── data/
│   ├── human_demonstrations/
│   └── autonomous_logs/
├── artifacts/
│   ├── checkpoints/
│   └── reports/
├── tests/
├── scripts/
└── docs/
    ├── architecture.md
    ├── project_overview.md
    ├── training_manual.md
    ├── algorithm_notes.md
    ├── ui_design_system.md
    ├── security_notes.md
    └── human_training_7_day_plan.md
```

---

## 4. Architecture guidelines (new)

Treat the system as five layers with one-directional dependencies:

```
envs  →  agents  →  training  →  simulation  →  UI (pages)
                data  ↗
```

- **envs**: HighwayEnv factory, state builder, shared action set. Knows
  nothing about agents or Streamlit.
- **agents**: DQN (R) and SARSA (S) as independent classes implementing the
  same small interface (`act`, `update`, `save`, `load`, `set_eval_mode`).
  Neither agent module imports the other.
- **data**: recording, validation, and loading of human demonstrations and
  autonomous logs. Pure functions where possible.
- **training**: orchestrates envs + agents + data. This is the only layer
  allowed to know about both R and S at once.
- **simulation**: a thin façade that Streamlit pages call — starts/steps the
  env, exposes current state for rendering, exposes metrics. Pages never
  import from `envs`, `agents`, or `training` directly.
- **UI (pages)**: rendering and interaction only. No training logic, no file
  I/O beyond what `simulation`/`data` expose.

Write `docs/architecture.md` in Build Step 1 and update it — not recreate it
from scratch — whenever a layer's public interface changes.

---

## 5. Security & robustness checklist (new — apply at every build step)

- [ ] Inputs from the UI are type-checked and range-checked before use
- [ ] File I/O only reads/writes inside `data/` and `artifacts/`, never outside
      the project root
- [ ] Human demonstration files are validated (schema + action set) before
      being trusted by any trainer
- [ ] No bare `except:`; exceptions are caught narrowly and logged
- [ ] No hardcoded absolute paths, usernames, or secrets
- [ ] Long-running loops (training, simulation) support clean interruption
      without corrupting saved checkpoints
- [ ] Dependencies pinned in `requirements.txt`

---

## 6. UI/UX design system (expanded)

Document this in `docs/ui_design_system.md` before building pages.

- **Palette**: a small, restrained set (e.g., dark asphalt background, one
  accent color for R, one distinct accent color for S, neutral grays for
  everything else). No unexplained gradients or glassmorphism.
- **Typography**: one UI font, one monospace font for numeric/telemetry
  readouts. No decorative fonts.
- **Iconography/imagery**: generate a small fixed set of simple, consistent
  icons/sprites once (top-down car icons for R and S in their accent colors,
  a lane/road texture, a favicon). Store them in `assets/` and reuse — do not
  pull stock photography or generate new art on every run.
- **Layout**: primary viewport for the road/simulation; a compact side rail
  for controls and live readouts (speed, lane, action, reward). Avoid stacks
  of generic "metric cards."
- **Motion**: use CSS transitions or canvas/SVG redraws for the live view;
  use animated Plotly/Altair traces for performance charts. Nothing should
  feel like a stock hero-image dashboard.

---

## 7. Streamlit pages (revised set)

1. **Simulation** — start/reset the env, pick R or S to view, step through
   manually or run autonomously at adjustable speed.
2. **Human Training** — keyboard-driven control of the selected vehicle,
   episode recording, save/discard controls, dataset summary.
3. **Live Tracking** *(new)* — a real-time top-down or side view of R and S
   on the road with smoothly animated positions, current speed, current
   action, and instantaneous reward, updating each simulation step without
   full-page reloads.
4. **Performance & Analytics** *(new)* — R vs S comparison: reward curves,
   collision rate, survival time, average speed, lane-change counts, action
   distribution, DQN loss, SARSA Q-value/update stats — as animated or
   incrementally-updating charts, not static screenshots.
5. **Train & Compare** — load human data, warm-start both agents, run
   autonomous training, trigger fixed-seed evaluation.
6. **Docs** *(new)* — renders `docs/project_overview.md`,
   `docs/algorithm_notes.md`, and `docs/architecture.md` inside the app so a
   reviewer never has to leave Streamlit to understand the project.

---

## 8. Build steps (same five phases, each now closes with a doc + checklist)

Keep the original Build Step 1–5 prompts and definitions of done from the
source plan. Append this closing requirement to **every** step:

> **Before marking this step done:**
> 1. Update the relevant file(s) in `docs/` describing what was built and why.
> 2. Confirm the architecture rule for this layer (§4) wasn't violated.
> 3. Run through the security & robustness checklist (§5) for any new file
>    I/O, user input, or external call introduced in this step.
> 4. If this step touched the UI, confirm it matches `docs/ui_design_system.md`.

Step-specific additions:

- **Step 1 (Environment + state)**: write the first version of
  `docs/architecture.md`.
- **Step 2 (Human recorder)**: write `docs/security_notes.md` covering file
  I/O and input validation for recorded episodes.
- **Step 3 (R + S learners)**: write `docs/algorithm_notes.md` explaining DQN
  and SARSA in plain language, and confirm neither agent module imports the
  other (architecture check).
- **Step 4 (Joint training + Streamlit)**: build the **Live Tracking** and
  **Performance & Analytics** pages as first-class pages, not an afterthought
  bolted onto Simulation; generate and check in the icon/sprite assets here.
- **Step 5 (QA + handoff)**: finalize all `docs/` files, add the **Docs** page,
  and re-run the full security & robustness checklist project-wide.

---

## 9. Handoff to the 7-day human-training plan

No change to the 7-day plan's daily cycle or evaluation rules — it still
starts only after Build Steps 1–5 (as revised above) are complete. One
addition: from Day 1 onward, use the **Live Tracking** page to drive and the
**Performance & Analytics** page to review each day's evaluation, so the
human-training loop and the UI stay the same tool throughout.