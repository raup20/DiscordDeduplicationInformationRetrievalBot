import discord
from discord.ext import commands
from memory_store import store_message, find_similar
from simhash_engine import simhash

TOKEN = "here needs to be inserted a token. right now its my personal bots token, looking into making it publicly available, in the meantime u can insert the token of your own bot for testing"

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f" Bot ready, {bot.user}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    user_msg = message.content
    print(f"Received: {user_msg}")
    h = simhash(user_msg)
    print(f"SimHash: {h}")

    similar = find_similar(user_msg)
    if similar:
        print("Similar found:")
        for s, d in similar[:3]:
            print(f"   -> '{s}' (distance {d})")

        best, score = similar[0]
        if best != user_msg:
            await message.channel.send(
                f"Similar previously asked:\n**{best}**\n(Hamming distance: {score})"
            )

    store_message(user_msg)
    await bot.process_commands(message)

bot.run(TOKEN)
