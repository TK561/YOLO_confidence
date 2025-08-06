"""
Configuration settings for YOLO detection
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Output settings
OUTPUT_DIR = os.getenv('OUTPUT_DIR', r'C:\Users\filqo\OneDrive\Desktop')

# Detection thresholds
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.25'))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', '0.45'))
DUPLICATE_IOU_THRESHOLD = float(os.getenv('DUPLICATE_IOU_THRESHOLD', '0.8'))

# Display settings
FONT_SIZE = float(os.getenv('FONT_SIZE', '3.5'))
FONT_THICKNESS = int(os.getenv('FONT_THICKNESS', '6'))
BOX_THICKNESS = int(os.getenv('BOX_THICKNESS', '12'))

# Model settings
MODEL_NAME = os.getenv('MODEL_NAME', 'yolov8n.pt')

# Debug mode
DEBUG_MODE = os.getenv('DEBUG_MODE', 'true').lower() == 'true'

# Color settings
RANDOM_SEED = 42  # For consistent colors across runs