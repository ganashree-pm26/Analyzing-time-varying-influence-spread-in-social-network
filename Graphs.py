import praw
import re
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# --- Reddit API Setup ---
reddit = praw.Reddit(
    client_id="b0Ppq9wrYTzYyLFNzZqjcw",
    client_secret="ybua2_kSmdQK_atcgCnbIyda0ixbfw",
    user_agent="TrendClusterGraph by Ganashree"
)

# --- Globals ---
topic_users = defaultdict(set)
G = nx.Graph()
word_freq = Counter()
stop_words = set(stopwords.words('english'))

# --- Config ---
TREND_KEYWORDS = ["ipl", "meta", "tesla", "modi", "ai", "chatgpt", "bjp", "congress", "usa", "ukraine"]  # Expand as needed

# --- Tokenization ---
def process_title(title):
    words = word_tokenize(title.lower())
    return [w for w in words if w.isalpha() and w not in stop_words]

# --- Update Graphs ---
def update(frame):
    ax.clear()

    # Clear previous graph
    G.clear()

    for topic, users in topic_users.items():
        # Only add if topic is active (>=2 users)
        if len(users) >= 2:
            for u1 in users:
                for u2 in users:
                    if u1 != u2:
                        G.add_node(u1, group=topic)
                        G.add_node(u2, group=topic)
                        G.add_edge(u1, u2, label=topic)

    colors = []
    for n in G.nodes:
        group = G.nodes[n].get("group", "")
        color = hash(group) % 10
        colors.append(color)

    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.tab10, ax=ax)
    ax.set_title("User Graphs per Trending Topic")

# --- Post Listener ---
def listen_reddit():
    for post in reddit.subreddit("all").stream.submissions(skip_existing=True):
        author = post.author.name if post.author else "unknown"
        title = post.title.lower()
        words = process_title(title)
        word_freq.update(words)

        for word in words:
            if word in TREND_KEYWORDS:
                topic_users[word].add(author)

# --- Plot Setup ---
fig, ax = plt.subplots(figsize=(12, 8))
ani = FuncAnimation(fig, update, interval=5000)

# --- Start Listener Thread ---
import threading
threading.Thread(target=listen_reddit, daemon=True).start()

# --- Show ---
plt.show()
