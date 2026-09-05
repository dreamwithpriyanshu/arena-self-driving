"""
Standalone PyGame script for real-time Human Training.
Run this instead of Streamlit for a 60 FPS driving experience!
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame
from src.human.episode_manager import EpisodeManager

def main():
    vehicle = sys.argv[1] if len(sys.argv) > 1 else "R"
    
    # We must initialize pygame to capture keyboard inputs properly
    pygame.init()
    
    print("=====================================================")
    print(f"🚗 Starting Real-Time Human Training for Vehicle {vehicle} 🚗")
    print("=====================================================")
    print("CONTROLS:")
    print(" - Arrow Keys to Drive")
    print(" - Space or Nothing to Coast/Idle")
    print(" - ESC to Quit & Discard")
    print("The episode will auto-save when it finishes (collision or timeout).")
    print("=====================================================\n")
    
    # Init the episode manager using the PyGame window renderer
    mgr = EpisodeManager(render_mode="human")
    mgr.start(vehicle=vehicle)
    
    try:
        while not mgr.is_done:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            
            # Check for quit
            if keys[pygame.K_ESCAPE]:
                print("ESC pressed. Discarding episode...")
                mgr.discard()
                pygame.quit()
                return
                
            # Map pygame keys to our KeyboardController string format
            action_key = " "
            if keys[pygame.K_LEFT]:
                action_key = "arrowleft"
            elif keys[pygame.K_RIGHT]:
                action_key = "arrowright"
            elif keys[pygame.K_UP]:
                action_key = "arrowup"
            elif keys[pygame.K_DOWN]:
                action_key = "arrowdown"
                
            # Step the environment (this handles internal physics rendering)
            mgr.act(action_key)
            
        print("\nEpisode finished!")
        path = mgr.save()
        print(f"✅ Successfully saved demonstration to: {path}")
        
    except KeyboardInterrupt:
        print("\nInterrupted! Discarding episode...")
        mgr.discard()
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
