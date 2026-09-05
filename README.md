# Self-Driving Car Simulation MVP

> **A comparative study of DQN vs SARSA for autonomous highway driving,
> trained on shared human demonstrations.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)]()
[![HighwayEnv](https://img.shields.io/badge/sim-HighwayEnv-green.svg)]()

---

## 🚗 What is this?

This project compares two reinforcement learning algorithms on the same
highway-driving task:

| Vehicle | Algorithm | Type |
|---------|-----------|------|
| **R** | DQN (Deep Q-Network) | Off-policy, neural network |
| **S** | SARSA | On-policy, tabular |

Both agents learn from **identical human demonstrations** collected in
[HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv).
The algorithm is the only controlled variable.

---

## 📁 Project structure

```
arena-self-driving/
├── app.py                    # Streamlit entry point
├── pages/                    # Streamlit multi-page UI
│   ├── 1_Simulation.py
│   ├── 2_Human_Training.py
│   ├── 3_Live_Tracking.py
│   ├── 4_Performance_Analytics.py
│   ├── 5_Train_and_Compare.py
│   └── 6_Docs.py
├── src/
│   ├── envs/                 # HighwayEnv factory, state builder, actions
│   ├── agents/               # DQN (R) and SARSA (S) implementations
│   ├── human/                # Keyboard control & episode recording
│   ├── data/                 # Demo recording, validation, loading
│   ├── training/             # Training orchestration
│   ├── evaluation/           # Fixed-seed evaluation & metrics
│   └── simulation/           # Façade for UI pages
├── configs/                  # YAML configuration files
├── assets/                   # Generated icons, sprites, textures
├── data/
│   ├── human_demonstrations/ # JSONL files from human driving
│   └── autonomous_logs/      # JSONL files from agent evaluation
├── artifacts/
│   ├── checkpoints/          # Saved model weights
│   └── reports/              # Evaluation reports
├── tests/                    # pytest test suite
├── scripts/                  # Utility & smoke-test scripts
└── docs/                     # Project documentation
```

---

## 🛠️ Setup

### Prerequisites

- Python **3.9 – 3.11**
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/dreamwithpriyanshu/arena-self-driving.git
cd arena-self-driving

# Create a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Environment variables

Copy and optionally edit the `.env` file.  It contains no secrets — only
local paths and runtime flags.

---

## 🚀 Quick start

### Run the Streamlit app

```bash
streamlit run app.py
```

### Run the smoke test (Build Step 1)

```bash
python scripts/test_step1.py
```

---

## 📊 Key metrics tracked

| Metric | Description |
|--------|------------|
| Mean episode reward | Average cumulative reward per episode |
| Collision rate | Fraction of episodes ending in a crash |
| Survival time | Average steps before episode ends |
| Average speed | Mean ego-vehicle speed |
| Lane-change count | Number of lane changes per episode |
| Action distribution | Frequency of each discrete action |
| DQN loss | Training loss for the DQN network |
| SARSA Q-value stats | Mean/max Q-values and update magnitudes |

---

## 🏗️ Build phases

| Step | Description | Status |
|------|-------------|--------|
| 1 | Environment, actions, state | 🔨 In progress |
| 2 | Human recorder | ⏳ Pending |
| 3 | R (DQN) + S (SARSA) agents | ⏳ Pending |
| 4 | Joint training + Streamlit UI | ⏳ Pending |
| 5 | QA + handoff | ⏳ Pending |

After all 5 build steps, a **7-day human training cycle** begins.

---

## 📖 Documentation

- [Architecture](docs/architecture.md) — layer diagram and dependency rules
- [Project Overview](docs/project_overview.md) — goals and tech stack
- [Algorithm Notes](docs/algorithm_notes.md) — DQN vs SARSA explained *(Step 3)*
- [Security Notes](docs/security_notes.md) — file I/O and input validation *(Step 2)*
- [UI Design System](docs/ui_design_system.md) — palette, typography, layout *(Step 4)*

---

## 📄 License

This project is for educational purposes.

---

*Built with [Antigravity](https://antigravity.dev) + HighwayEnv + Streamlit*
