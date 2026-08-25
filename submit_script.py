import subprocess
import os

title = "⚡ [Performance] Replace PIL with cv2.imdecode for image decoding"
with open('pr_description.md', 'r') as f:
    body = f.read()

# Since I am executing this programmatically inside the bash session to simulate a submit,
# the tool to be used here is the actual MCP tool 'submit', but in python I'll just write it for the agent context.
print(f"Title: {title}")
print(f"Body:\n{body}")
