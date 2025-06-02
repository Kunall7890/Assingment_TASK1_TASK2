import os
import time
import logging
from src.run_pipeline import run_pipeline
from src.visualize import Visualizer
import gradio as gr
from src.app import create_interface
import cv2
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def record_demo_video(duration: int = 300, output_path: str = 'demo_video.mp4'):
    """
    Record a demo video of the application.
    
    Args:
        duration (int): Duration of the video in seconds
        output_path (str): Path to save the video
    """
    try:
        # Initialize video writer
        screen_width = 1920
        screen_height = 1080
        fps = 30
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (screen_width, screen_height))
        
        # Start recording
        logger.info(f"Recording demo video for {duration} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Capture screen
            screen = np.array(ImageGrab.grab(bbox=(0, 0, screen_width, screen_height)))
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(screen)
            
            # Add timestamp
            current_time = datetime.now().strftime("%H:%M:%S")
            cv2.putText(
                screen,
                current_time,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            # Wait for next frame
            time.sleep(1/fps)
        
        # Release video writer
        out.release()
        logger.info(f"Demo video saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error recording demo video: {str(e)}")
        raise

def main():
    """Main function to run the pipeline and create demo."""
    try:
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        
        # Step 1: Run the pipeline
        logger.info("Starting the machine learning pipeline...")
        run_pipeline()
        
        # Step 2: Generate visualizations
        logger.info("Generating visualizations...")
        visualizer = Visualizer(
            metrics_path='models/metrics.json',
            entity_stats_path='data/entity_statistics.json'
        )
        visualizer.generate_all_visualizations()
        
        # Step 3: Launch the Gradio interface
        logger.info("Launching the Gradio interface...")
        interface = create_interface()
        
        # Step 4: Record demo video
        logger.info("Recording demo video...")
        record_demo_video()
        
        logger.info("Demo creation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in demo creation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 