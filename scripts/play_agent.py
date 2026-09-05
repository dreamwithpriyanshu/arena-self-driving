"""
Standalone PyGame script for real-time Agent Evaluation.
Watch the trained DQN or SARSA agent drive autonomously at 60 FPS!
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame
from src.agents.dqn import DQNAgent
from src.agents.sarsa import SARSAAgent
from src.training.orchestrator import TrainingOrchestrator
from src.simulation.env_manager import EnvManager

def main():
    vehicle = sys.argv[1].upper() if len(sys.argv) > 1 else "R"
    
    if vehicle not in ("R", "S"):
        print("Vehicle must be 'R' (DQN) or 'S' (SARSA).")
        sys.exit(1)
        
    print("=====================================================")
    print(f"🤖 Starting Real-Time Agent Evaluation for Vehicle {vehicle} 🤖")
    print("=====================================================")
    print("Watch the agent drive autonomously.")
    print("Press ESC or close the window to quit.")
    print("=====================================================\n")

    # We must initialize pygame to capture keyboard inputs for quitting
    pygame.init()
    
    # Init agents and orchestrator
    dqn = DQNAgent()
    sarsa = SARSAAgent()
    orchestrator = TrainingOrchestrator(dqn, sarsa)
    
    # Attempt to load checkpoints
    try:
        orchestrator.load_checkpoints()
        print("✅ Checkpoints loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Failed to load checkpoints. Agent may be untrained. ({e})")
        
    env_mgr = EnvManager(render_mode="human")
    
    try:
        # evaluate_episode runs the loop for us. We pass a step callback 
        # to process pygame events so the window doesn't freeze and can be closed.
        def step_callback(result, metrics):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                raise KeyboardInterrupt
                
        summary = orchestrator.evaluate_episode(
            vehicle=vehicle,
            env_mgr=env_mgr,
            step_callback=step_callback
        )
        
        print("\nEpisode finished!")
        print(f"Summary: {summary}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user! Stopping evaluation.")
    finally:
        env_mgr.close()
        pygame.quit()

if __name__ == "__main__":
    main()
