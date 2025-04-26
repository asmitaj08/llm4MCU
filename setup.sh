#!/bin/bash

# Exit the script on any error
set -e

# Step 1: Clone the cmsis-svd-data repository into the root directory
echo "Cloning cmsis-svd-data repository..."
git clone https://github.com/cmsis-svd/cmsis-svd-data.git cmsis_svd/data
echo "Repository cloned successfully."

# Step 2: Install the required Python packages
echo "Installing required Python packages..."
pip install -r requirements.txt
echo "Requirements installed successfully."

# Step 3: Run the generate_dataset.py script
echo "Running the generate_dataset.py script..."
python generate_dataset.py
echo "Dataset generation completed successfully."

echo "Setup completed. You are ready to go!"
