import sys
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the SadTalker directory to the Python path
sadtalker_dir = os.path.join(current_dir, 'src', 'SadTalker')
sys.path.append(sadtalker_dir)

# Add the SadTalker's src directory to the Python path
sadtalker_src_dir = os.path.join(sadtalker_dir, 'src')
sys.path.append(sadtalker_src_dir)
