"""Evaluate the SARSA checkpoint and write a compact performance timeline."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.sarsa import SARSAAgent
from src.simulation.env_manager import EnvManager
from src.storage import checkpoints_dir, PROJECT_ROOT, ensure_storage_dirs
from src.training.orchestrator import TrainingOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "accuracy.md")
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        raise SystemExit("episodes and steps must be positive")

    ensure_storage_dirs()
    agent = SARSAAgent()
    agent.load(checkpoints_dir())
    agent.set_eval_mode(True)
    orchestrator = TrainingOrchestrator(agent, checkpoint_dir=checkpoints_dir())
    env = EnvManager(render_mode=None, config_overrides={"duration": args.steps})
    results = []
    try:
        for index in range(args.episodes):
            results.append(orchestrator.evaluate_episode(env, seed=args.seed + index, max_steps=args.steps))
    finally:
        env.close()

    rewards = [float(item["total_reward"]) for item in results]
    steps = [int(item["steps"]) for item in results]
    collision_free = sum(not item["terminated"] for item in results)
    report = args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with report.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n| {timestamp} | {args.episodes} | {statistics.mean(rewards):.3f} | "
            f"{statistics.mean(steps):.1f} | {collision_free / args.episodes:.1%} |\n"
        )
    print(f"Average reward: {statistics.mean(rewards):.3f}")
    print(f"Average survival steps: {statistics.mean(steps):.1f}")
    print(f"Collision-free rate: {collision_free / args.episodes:.1%}")


if __name__ == "__main__":
    main()
