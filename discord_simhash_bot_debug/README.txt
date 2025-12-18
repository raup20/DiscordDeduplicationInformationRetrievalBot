# Running the Bot
## 1. Install dependencies

Make sure you are using Python 3.10+ (recommended).

pip install -r requirements.txt


This installs:

discord.py – Discord bot framework

sentence-transformers – embedding model for semantic similarity

numpy – vector operations

## 2. Set the Discord bot token

Set your bot token as an environment variable:

PowerShell (Windows):

$env:DISCORD_TOKEN="YOUR_BOT_TOKEN"


Linux / macOS:

export DISCORD_TOKEN="YOUR_BOT_TOKEN"

## 3. Run the bot

From the project directory:

python bot.py

What the bot does

Listens to messages in Discord channels

Embeds each message into a semantic vector space

Classifies message intent (question / answer / other)

When a new question is asked:

Searches for semantically similar past questions

Retrieves and suggests the best linked answer

When an answer is sent:

Links it to the most relevant recent question

## Debug output

The bot prints detailed debug information to the console:

intent classification scores

similarity values

answer–question linking decisions

This is useful for understanding and evaluating system behavior.