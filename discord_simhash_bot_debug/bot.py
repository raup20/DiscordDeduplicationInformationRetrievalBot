import os

import discord
from discord.ext import commands

from intent_classifier import IntentClassifier
from qa_store import QAStore

TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

store = QAStore(dim=384, n_planes=64, n_bands=8)
intent = IntentClassifier()

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    user_msg = (message.content or "").strip()
    if not user_msg:
        await bot.process_commands(message)
        return

    # If the user replied to a specific message, that's strong evidence it's an answer.
    reply_to = None
    if message.reference and message.reference.message_id:
        reply_to = int(message.reference.message_id)

    label = "other"
    conf = 0.0

    if reply_to is not None:
        label = "answer"
        conf = 1.0
    else:
        res = intent.classify(user_msg)
        label = res.label
        conf = res.confidence

    # Debug prints (keep during development; can remove later)
    print(f"[{message.channel.id}] {message.author.id} intent={label} conf={conf:.2f} :: {user_msg}")

    if label == "question":
        # Retrieve similar previous question(s)
        hits = store.search_questions(user_msg, top_k=3, min_sim=0.78)

        if hits:
            best_q, sim = hits[0]

            # Don't trigger on identical message text (paste/repost)
            if best_q.text.strip() != user_msg.strip():
                ans = store.get_best_answer(best_q)

                if ans:
                    await message.channel.send(
                        "🔍 **Similar previously asked:**\n"
                        f"**Q:** {best_q.text}\n"
                        f"**A:** {ans}\n"
                        f"(cosine similarity: {sim:.3f})"
                    )
                else:
                    await message.channel.send(
                        "🔍 **Similar previously asked:**\n"
                        f"**{best_q.text}**\n"
                        f"(cosine similarity: {sim:.3f})"
                    )

        # Store the new question
        store.add_question(
            msg_id=message.id,
            channel_id=message.channel.id,
            author_id=message.author.id,
            text=user_msg,
            ts=message.created_at.timestamp(),
        )

    elif label == "answer":
        # Store the answer and try to link it to a question.
        store.add_answer(
            msg_id=message.id,
            channel_id=message.channel.id,
            author_id=message.author.id,
            text=user_msg,
            reply_to_msg_id=reply_to,
            ts=message.created_at.timestamp(),
        )

    else:
        # ignore casual chatter (keeps DB clean)
        pass

    await bot.process_commands(message)

if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError("DISCORD_TOKEN env var is missing. Set it and restart the bot.")
    bot.run(TOKEN)
