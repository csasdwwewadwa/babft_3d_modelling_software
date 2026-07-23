import numpy as np
from scipy.spatial.transform import Rotation
import math
import json
import keyboard
from tqdm import tqdm

import io
import json
import urllib.request
import pyautogui

from macro import Vector2, Vector3, Workspace, Cuboid, Color


DISCORD_WEBHOOK = "https://discordapp.com/api/webhooks/1529215287642034206/MlOr1nBhTWXrr-MBsNHJzvgQxCzDyF5xUoqFhEIlvfOPLkX0fDFCMA05Ih8GKPW5r9F0"
DISCORD_PING_CSA = "<@727017345356398592>"

def report_message(message: str) -> None:
    """Sends a plain text message to the Discord webhook."""
    payload = json.dumps({"content": message}).encode("utf-8")

    req = urllib.request.Request(
        DISCORD_WEBHOOK,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "PyAutoGUI-Reporter/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as response:
            pass
    except Exception as e:
        print(f"Failed to send message to Discord: {e}")

def report_screenshot(message: str) -> None:
    """Takes an in-memory screenshot via PyAutoGUI and posts it with a message to Discord."""
    # 1. Take screenshot and save directly to memory buffer (PNG)
    img = pyautogui.screenshot()
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    # 2. Build multipart/form-data payload
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = io.BytesIO()

    # Append JSON payload field
    payload_json = json.dumps({"content": message})
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(
        b'Content-Disposition: form-data; name="payload_json"\r\nContent-Type: application/json\r\n\r\n'
    )
    body.write(payload_json.encode("utf-8"))
    body.write(b"\r\n")

    # Append image file field
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(
        b'Content-Disposition: form-data; name="files[0]"; filename="screenshot.png"\r\nContent-Type: image/png\r\n\r\n'
    )
    body.write(img_bytes)
    body.write(b"\r\n")

    # End boundary
    body.write(f"--{boundary}--\r\n".encode("utf-8"))

    # 3. Post request
    req = urllib.request.Request(
        DISCORD_WEBHOOK,
        data=body.getvalue(),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "PyAutoGUI-Reporter/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as response:
            pass
    except Exception as e:
        print(f"Failed to send screenshot to Discord: {e}")


def load_cuboids(file_path:str):
    with open(file_path) as f:
        a = json.load(f)
    cuboids:list[Cuboid] = []
    for v in a:
        cuboids.append(Cuboid(
            Vector3(*v['position']), 
            Rotation.from_euler('xyz', v['rotation']), 
            Vector3(*v['scale']), 
            Color(*v['color']) if v['color'] else None
            ))
    cuboids.sort(key=lambda v: v.position.x)
    cuboids.sort(key=lambda v: v.position.z)
    cuboids.sort(key=lambda v: v.position.y)
    return cuboids


stop_program = False

def stop_trigger():
    global stop_program
    tqdm.write("\n[Stop signal received] Finishing up or stopping before the next block...")
    stop_program = True

# This listens in the background. As soon as "=" is pressed, it calls stop_trigger
keyboard.add_hotkey('f7', stop_trigger)



workspace = Workspace() 
cuboids = load_cuboids(r'files/cirno.json')

progress = -1
with open('progress.txt') as f:
    progress = int(f.read())

print(f'press "f6" to init and start, f7 to stop. progress: {progress}')
keyboard.wait('f6')

report_message('# STARTING ----------')

workspace.init()

def save_faulty_block_id(id:int, reason:str):
    with open('faulty_block_id.txt', 'a') as f:
        f.write(f'{id} - {reason}\n')


for i in tqdm(range(progress+1, len(cuboids)), initial=progress+1, total=len(cuboids)):
    
    
    cuboid = cuboids[i]
 
    # Check if the stop flag was set before starting the next block
    if stop_program:
        tqdm.write("Program stopped by user.")
        break

    # print('press "-" to continue')
    # keyboard.wait('-')
    try:
        workspace.build_block(cuboid)
    except Exception as e:
        error_message = str(e)
        report_message(f'{error_message}')
        match error_message:
            case msg if 'in-game scaling knob\'s location is wrong' in msg:
                tqdm.write(f'[{i}::WARN] - {msg}')
                save_faulty_block_id(i, msg)
                continue
            case _:
                report_screenshot(f'CRITICAL ERROR, EXECUTION TERMINATED! {DISCORD_PING_CSA}')
                raise

    with open('progress.txt', 'w') as f:
        f.write(str(i))

    if i%100 == 0:
        workspace.save(f'cirno {i}')
        report_screenshot(f'saved: {i}')
