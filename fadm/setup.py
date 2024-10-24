import os
import subprocess

## Download comfy ui + extensions
os.makedirs("external", exist_ok=True)
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)
subprocess.run("git clone https://github.com/comfyanonymous/ComfyUI", cwd="./external/", shell=True)
subprocess.run("git clone https://github.com/Fannovel16/comfyui_controlnet_aux/", cwd="./external/ComfyUI/custom_nodes/", shell=True)

## TODO automatic conda stuff
# subprocess.run("git reset --hard 0dccb4617de61b81763321f01ae527dbe3b01202",cwd="./external/", shell=True)
# subprocess.run("git reset --hard ac32b6e826da542f84d615c66f316a9f3c176d96", cwd="./external/ComfyUI/custom_nodes/", shell=True)