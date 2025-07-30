import logging
import os
from datetime import datetime

# Step 1: Create logs directory (only the folder, not full path)
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

# Step 2: Define log file name and full path
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# Step 3: Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def get_log_path():
    return LOG_FILE_PATH
