"""
Automated Screenshot Capture for Streamlit MVP.
Uses Playwright to headlessly navigate the Streamlit app and save screenshots.
"""

import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    target_dir = Path("tutorials/screenshots")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    pages = [
        ("Simulation", "1_simulation.png"),
        ("Human Training", "2_human_training.png"),
        ("Live Tracking", "3_live_tracking.png"),
        ("Performance Analytics", "4_performance_analytics.png"),
        ("Train and Compare", "5_train_and_compare.png"),
        ("Docs", "6_docs.png")
    ]
    
    print("Starting Playwright capture...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Use a high-res window size
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        
        # Go to the root app
        page.goto("http://localhost:8501")
        
        # Wait for Streamlit to fully boot and render
        page.wait_for_selector('div.stApp', timeout=15000)
        time.sleep(3) # Let fonts/CSS settle
        
        for nav_text, filename in pages:
            print(f"Navigating to {nav_text}...")
            # Click the sidebar link
            try:
                page.click(f'a:has-text("{nav_text}")', timeout=5000)
                # Wait for the main title of the page to update/load
                time.sleep(2)
                
                # Take the screenshot
                out_path = target_dir / filename
                page.screenshot(path=str(out_path))
                print(f"  -> Saved {filename}")
            except Exception as e:
                print(f"  -> Failed to capture {nav_text}: {e}")
                
        browser.close()
        
    print("Capture complete!")

if __name__ == "__main__":
    main()
