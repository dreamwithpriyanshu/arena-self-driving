"""Headless, reproducible trainer for the single tabular SARSA policy."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.sarsa import SARSAAgent
from src.simulation.env_manager import EnvManager
from src.training.orchestrator import TrainingOrchestrator
from src.storage import checkpoints_dir, demonstrations_dir, artifacts_dir, ensure_storage_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless trainer for the tabular SARSA driving policy")
    parser.add_argument("--episodes", "--episode-count", dest="episodes", type=int, default=50)
    parser.add_argument("--warm-start", action="store_true", help="Learn from recorded arrow-key demonstrations first.")
    parser.add_argument("--resume", action="store_true", help="Load the SARSA Q-table before running.")
    parser.add_argument("--evaluation-only", action="store_true", help="Evaluate the Q-table without updates.")
    parser.add_argument("--seed", type=int, help="Seed Python, NumPy, and each episode.")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--greedy", action="store_true", help="Use a zero-exploration policy.")
    parser.add_argument("--vehicles-count", type=int, default=15)
    parser.add_argument("--vehicles-density", type=float, default=1.0)
    parser.add_argument("--duration", type=int, default=300, help="Maximum decisions per episode.")
    parser.add_argument("--simulation-frequency", type=int, default=15)
    parser.add_argument("--policy-frequency", type=int, default=5)
    parser.add_argument("--collision-reward", type=float, default=-2.0)
    parser.add_argument("--right-lane-reward", type=float, default=0.1)
    parser.add_argument("--high-speed-reward", type=float, default=0.4)
    parser.add_argument("--lane-change-reward", type=float, default=-0.05)
    parser.add_argument("--save-freq", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=demonstrations_dir())
    parser.add_argument("--checkpoint-dir", type=Path, default=checkpoints_dir())
    parser.add_argument("--history-dir", type=Path, default=artifacts_dir())
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = build_parser().parse_args()
    ensure_storage_dirs()
    if min(args.episodes, args.duration, args.save_freq, args.log_interval, args.simulation_frequency, args.policy_frequency) < 1:
        raise SystemExit("Episode, frequency, duration, save, and log values must be positive.")
    if args.policy_frequency > args.simulation_frequency:
        raise SystemExit("--policy-frequency cannot exceed --simulation-frequency.")
    if args.seed is not None:
        seed_everything(args.seed)
    if args.greedy or args.evaluation_only:
        args.epsilon_start, args.epsilon_end, args.epsilon_decay = 0.0, 0.0, 1.0

    agent = SARSAAgent(lr=args.lr, gamma=args.gamma, epsilon_start=args.epsilon_start,
                       epsilon_end=args.epsilon_end, epsilon_decay=args.epsilon_decay)
    orchestrator = TrainingOrchestrator(sarsa_agent=agent, checkpoint_dir=args.checkpoint_dir)
    if args.resume or args.evaluation_only:
        try:
            orchestrator.load_checkpoints()
            print(f"Loaded SARSA Q-table from {args.checkpoint_dir}.")
        except FileNotFoundError as error:
            raise SystemExit(f"Cannot resume/evaluate: {error}") from error
    if args.warm_start and not args.evaluation_only:
        print(f"Warm start: {orchestrator.warm_start(args.data_dir)}")

    args.history_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{time.time_ns() % 1_000_000_000:09d}"
    history_path = args.history_dir / f"training_history_{run_id}.jsonl"
    meta_path = args.history_dir / f"training_history_{run_id}.meta.json"
    metadata = {"run_id": run_id, "history_path": str(history_path), "created_at": time.time(),
                "mode": "evaluation" if args.evaluation_only else "training", "algorithm": "tabular_sarsa",
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    env_mgr = EnvManager(render_mode=None, config_overrides={
        "vehicles_count": args.vehicles_count, "vehicles_density": args.vehicles_density,
        "duration": args.duration, "simulation_frequency": args.simulation_frequency,
        "policy_frequency": args.policy_frequency, "collision_reward": args.collision_reward,
        "right_lane_reward": args.right_lane_reward, "high_speed_reward": args.high_speed_reward,
        "lane_change_reward": args.lane_change_reward,
    })
    run_episode = orchestrator.evaluate_episode if args.evaluation_only else orchestrator.train_episode
    started_at = time.time()
    try:
        with history_path.open("x", encoding="utf-8") as history_file:
            for episode_number in range(1, args.episodes + 1):
                episode_seed = None if args.seed is None else args.seed + episode_number - 1
                metrics = run_episode(env_mgr=env_mgr, seed=episode_seed, max_steps=args.duration)
                record = {"run_id": run_id, "episode_num": episode_number, "episode_seed": episode_seed,
                          "mode": metadata["mode"], "timestamp": time.time(), **metrics}
                history_file.write(json.dumps(record) + "\n")
                history_file.flush()
                if episode_number % args.log_interval == 0:
                    print(f"SARSA episode {episode_number}/{args.episodes}: reward={metrics['total_reward']:+.2f}, steps={metrics['steps']}")
                if not args.evaluation_only and episode_number % args.save_freq == 0:
                    orchestrator.save_checkpoints()
    finally:
        env_mgr.close()
        if not args.evaluation_only:
            orchestrator.save_checkpoints()
    print(f"{metadata['mode'].title()} finished in {time.time() - started_at:.2f}s.")
    print(f"History: {history_path}\nMetadata: {meta_path}\nCheckpoint: {args.checkpoint_dir / 'sarsa_q_table.npy'}")


if __name__ == "__main__":
    main()
