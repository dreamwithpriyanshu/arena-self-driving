"""
The Ultimate CLI Trainer.
Headless orchestration for DQN and SARSA agents. 
Supports batch training, warm-starting, and live metrics logging.
"""
import sys
import argparse
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.dqn import DQNAgent
from src.agents.sarsa import SARSAAgent
from src.training.orchestrator import TrainingOrchestrator
from src.simulation.env_manager import EnvManager

def main():
    parser = argparse.ArgumentParser(description="Headless CLI Trainer for DQN and SARSA")
    parser.add_argument("--agent", type=str, choices=["R", "S", "BOTH"], default="BOTH", help="Agent(s) to train: R (DQN), S (SARSA), or BOTH")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to train")
    parser.add_argument("--warm-start", action="store_true", help="Warm-start agents using human demonstrations before training")
    parser.add_argument("--batch-size", type=int, default=64, help="DQN replay buffer batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--target-update-freq", type=int, default=100, help="Target network update frequency (steps)")
    parser.add_argument("--buffer-capacity", type=int, default=10000, help="Replay buffer capacity for DQN")
    parser.add_argument("--device", type=str, choices=["auto","cpu","cuda"], default="auto", help="Device to run training on")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Exploration decay rate")
    parser.add_argument("--vehicles-count", type=int, default=15, help="Number of NPC vehicles on the road")
    parser.add_argument("--duration", type=int, default=120, help="Max duration of an episode in steps")
    parser.add_argument("--save-freq", type=int, default=5, help="Save checkpoints every N episodes")
    parser.add_argument("--log-interval", type=int, default=1, help="Print metrics every N episodes")
    parser.add_argument("--history-mode", type=str, choices=["per-run","append"], default="per-run", help="How to persist training history: per-run timestamped file or append to single file")
    parser.add_argument("--history-dir", type=str, default="artifacts", help="Directory to store training history files")
    parser.add_argument("--greedy", action="store_true", help="Run training with greedy policy (epsilon=0) — useful for deterministic runs/evaluation")
    
    args = parser.parse_args()

    # Apply greedy flag: force epsilon to 0 and disable decay
    if args.greedy:
        args.epsilon_start = 0.0
        args.epsilon_decay = 1.0

    # Device selection (allow override)
    if args.device == "auto":
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=======================================")
    print("      SELF-DRIVING CAR CLI TRAINER")
    print("=======================================")
    print(f"Agents       : {args.agent}")
    print(f"Episodes     : {args.episodes}")
    print(f"Warm Start   : {'Yes' if args.warm_start else 'No'}")
    print(f"Epsilon      : {args.epsilon_start} (Decay: {args.epsilon_decay})")
    print(f"Traffic      : {args.vehicles_count} NPCs")
    print("=======================================")

    # Initialize Agents
    dqn = DQNAgent(
        batch_size=args.batch_size,
        epsilon_start=args.epsilon_start,
        epsilon_decay=args.epsilon_decay
    )
    sarsa = SARSAAgent(
        epsilon_start=args.epsilon_start,
        epsilon_decay=args.epsilon_decay
    )
    
    checkpoint_dir = Path("artifacts/checkpoints")
    
    # Try loading existing checkpoints
    try:
        dqn.load(checkpoint_dir)
        print("Loaded existing DQN checkpoint.")
    except FileNotFoundError:
        print("No existing DQN checkpoint. Starting fresh.")
        
    try:
        sarsa.load(checkpoint_dir)
        print("Loaded existing SARSA checkpoint.")
    except FileNotFoundError:
        print("No existing SARSA checkpoint. Starting fresh.")

    # Orchestrator
    orchestrator = TrainingOrchestrator(dqn_agent=dqn, sarsa_agent=sarsa, checkpoint_dir=checkpoint_dir)

    # Warm Start
    if args.warm_start:
        stats = orchestrator.warm_start()
        print(f"\nWarm-Start Complete:")
        print(f"  Transitions Loaded  : {stats.get('loaded', 0)}")
        print(f"  DQN Buffer Prefilled: {stats.get('r_prefilled', 0)}")
        print(f"  SARSA Table Updates : {stats.get('s_warm_started', 0)}")

    # Prepare training history file according to the requested mode
    history_dir = Path(args.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.history_mode == "per-run":
        run_id = ts
        history_path = history_dir / f"training_history_{run_id}.jsonl"
    else:
        # append to a single canonical history file for backward compatibility
        # still create a per-run metadata file so runs are identifiable
        run_id = ts
        history_path = history_dir / "training_history.jsonl"

    # Persist run metadata alongside the history file for observability
    try:
        import json
        meta_path = history_dir / f"training_history_{run_id}.meta.json"
        meta = {
            "run_id": run_id,
            "history_mode": args.history_mode,
            "history_path": str(history_path),
            "args": {k: v for k, v in vars(args).items()},
            "created_at": time.time(),
        }
        with meta_path.open("w", encoding="utf-8") as mh:
            json.dump(meta, mh, indent=2)
    except Exception as e:
        print(f"[!] Warning: failed to write run metadata: {e}")

    # Environment
    config_overrides = {
        "vehicles_count": args.vehicles_count,
        "duration": args.duration,
    }
    env_mgr = EnvManager(render_mode="rgb_array", config_overrides=config_overrides)
    
    agents_to_train = ["R", "S"] if args.agent == "BOTH" else [args.agent]
    
    print("\nStarting Autonomous Training...")
    start_time = time.time()
    
    try:
        for ep in range(1, args.episodes + 1):
            print(f"\n--- Episode {ep}/{args.episodes} ---")
            
            for vehicle in agents_to_train:
                print(f"Training Agent {vehicle}...", end="", flush=True)
                
                metrics = orchestrator.train_episode(
                    vehicle=vehicle,
                    env_mgr=env_mgr,
                    max_steps=args.duration
                )
                
                print(f" Done! (Reward: {metrics['total_reward']:+.2f}, Steps: {metrics['steps']})")
                print(f"  Details: Epsilon={metrics.get('epsilon', 0):.3f} | Crashed: {metrics['terminated']}")

                # Persist per-episode metrics to a newline-delimited JSON file so the
                # Streamlit dashboard can reflect training progress after the run.
                try:
                    import json
                    record = {"episode_num": ep, "vehicle": vehicle, **metrics, "timestamp": time.time()}
                    with history_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")
                except Exception as e:
                    print(f"[!] Warning: failed to write training history: {e}")
                
            if ep % args.save_freq == 0:
                orchestrator.save_checkpoints()
                print(f"[!] Saved Checkpoints at Episode {ep}")
                
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    finally:
        env_mgr.close()
        orchestrator.save_checkpoints()
        
    end_time = time.time()
    print(f"\n=======================================")
    print(f"Training Finished! Total Time: {(end_time - start_time):.2f}s")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=======================================")


if __name__ == "__main__":
    main()
