import io
import json
import math
import asyncio
import numpy as np
from scipy.spatial.transform import Rotation
import pyautogui
from tqdm import tqdm
import discord
from discord.ext import commands

from macro import Vector2, Vector3, Workspace, Cuboid, Color

# ================= CONFIGURATION =================
BOT_TOKEN = "YOUR_DISCORD_BOT_TOKEN_HERE"  # Replace with your Bot Token
DISCORD_PING_CSA = "<@727017345356398592>"
# =================================================

# Setup Bot Intents
intents = discord.Intents.default()
intents.message_content = True  # Required to read text commands like /start
bot = commands.Bot(command_prefix="/", intents=intents)

# State variables
is_running = False
build_task = None

def load_cuboids(file_path: str):
    with open(file_path) as f:
        a = json.load(f)
    cuboids: list[Cuboid] = []
    for v in a:
        cuboids.append(
            Cuboid(
                Vector3(*v['position']), 
                Rotation.from_euler('xyz', v['rotation']), 
                Vector3(*v['scale']), 
                Color(*v['color']) if v['color'] else None
            )
        )
    cuboids.sort(key=lambda v: v.position.x)
    cuboids.sort(key=lambda v: v.position.z)
    cuboids.sort(key=lambda v: v.position.y)
    return cuboids

def save_faulty_block_id(block_id: int, reason: str):
    with open('faulty_block_id.txt', 'a') as f:
        f.write(f'{block_id} - {reason}\n')

async def send_screenshot(ctx, message: str):
    """Captures screenshot in-memory and sends directly to Discord channel."""
    img = pyautogui.screenshot()
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    
    file = discord.File(fp=img_buffer, filename="screenshot.png")
    await ctx.send(content=message, file=file)

async def build_loop(ctx):
    """Background task containing the main macro building loop."""
    global is_running

    workspace = Workspace() 
    cuboids = load_cuboids(r'files/cirno.json')

    progress = -1
    try:
        with open('progress.txt') as f:
            progress = int(f.read())
    except FileNotFoundError:
        progress = -1

    await ctx.send(f'# STARTING ----------\nProgress starting at index: `{progress + 1}`')
    workspace.init()

    for i in tqdm(range(progress + 1, len(cuboids)), initial=progress + 1, total=len(cuboids)):
        # Allow asyncio context switches so Discord bot remains responsive
        await asyncio.sleep(0.01)

        # Check for stop signal
        if not is_running:
            tqdm.write("\n[Stop signal received] Stopping macro execution.")
            await ctx.send("🛑 **Macro loop successfully stopped.**")
            return

        cuboid = cuboids[i]

        try:
            workspace.build_block(cuboid)
        except Exception as e:
            error_message = str(e)
            await ctx.send(f"⚠️ **Error at block {i}:** `{error_message}`")
            
            if 'in-game scaling knob\'s location is wrong' in error_message:
                tqdm.write(f'[{i}::WARN] - {error_message}')
                save_faulty_block_id(i, error_message)
                continue
            else:
                await send_screenshot(ctx, f'CRITICAL ERROR, EXECUTION TERMINATED! {DISCORD_PING_CSA}')
                is_running = False
                return

        with open('progress.txt', 'w') as f:
            f.write(str(i))

    await ctx.send("✅ **Building finished successfully!**")
    is_running = False


# ================= DISCORD BOT COMMANDS =================

@bot.event
async def on_ready():
    print(f"Bot connected as {bot.user.name} ({bot.user.id})")
    print("Ready to listen for /start or /stop commands...")

@bot.command(name="start")
async def start_command(ctx):
    global is_running, build_task

    if is_running:
        await ctx.send("⚠️ Macro is already running!")
        return

    is_running = True
    await ctx.send("🚀 Starting macro task...")
    
    # Run the building loop in the background without blocking the bot
    build_task = asyncio.create_task(build_loop(ctx))

@bot.command(name="stop")
async def stop_command(ctx):
    global is_running

    if not is_running:
        await ctx.send("⚠️ Macro is not currently running.")
        return

    is_running = False
    await ctx.send("⏳ Stopping macro after current block completes...")

if __name__ == "__main__":
    bot.run(BOT_TOKEN)