import os
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the output directory relative to the base directory
output_dir = os.path.join(base_dir, "output")
image_loc = os.path.join(base_dir, "src", "doctor1.jpeg")
print(output_dir)