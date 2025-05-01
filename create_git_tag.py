
# Script to create and push a git tag on host

import json
import subprocess
import argparse

# Parsing for optional capture push flag
parser = argparse.ArgumentParser(description="Create and push a git tag based on model release info")
parser.add_argument("--push", action="store_true", help="Push the tag to remote after creation")
args = parser.parse_args()

# Load model info
with open("model_release_info.json", 'r') as f:
    info = json.load(f)
model_name = info["model_name"]
version = info["version"]
tag_name = f"model-v{version}"

print(f"Creating git tag for {model_name} version {version}")
try:
    # Create tag depends on model info 
    subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Model release v{version}"], check=True)
    print(f"Created tag: {tag_name}")
    
    # Push tag if requested
    if args.push:
        print(f"Pushing tag to origin...")
        subprocess.run(["git", "push", "origin", tag_name], check=True)
        print(f"Successfully pushed tag {tag_name} to origin")
    else:
        print(f"Tag created but not pushed. To push: git push origin {tag_name}")
        
except subprocess.CalledProcessError as e:
    print(f"Error executing git command: {e}")
    print(f"Manual commands:")
    print(f"  Create: git tag -a {tag_name} -m \"Model release v{version}\"")
    print(f"  Push: git push origin {tag_name}")
except Exception as e:
    print(f"Unexpected error: {e}")
