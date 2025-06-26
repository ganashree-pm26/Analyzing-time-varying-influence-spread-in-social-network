import snscrape.modules.twitter as sntwitter
import networkx as nx
import pandas as pd
import time
import matplotlib.pyplot as plt

def add_tweet_to_graph(tweet):
    user = tweet.user.username
    mentions = [mention.username for mention in tweet.mentionedUsers or []]

    G.add_node(user)

    if not mentions:
        G.add_node("temp_user")
        G.add_edge(user, "temp_user", timestamp=str(tweet.date))
    else:
        for mention in mentions:
            G.add_node(mention)
            G.add_edge(user, mention, timestamp=str(tweet.date))


def draw_graph():
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5)  # Layout for node positions
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    plt.title("Live Influence Graph from Twitter", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Create an empty graph
G = nx.DiGraph()

# Choose your hashtag or topic
HASHTAG = "#AI"

def add_tweet_to_graph(tweet):
    user = tweet.user.username
    mentions = [mention.username for mention in tweet.mentionedUsers or []]

    # Add the tweet's author
    G.add_node(user)

    # Add mentioned users as nodes and create edges
    for mention in mentions:
        G.add_node(mention)
        G.add_edge(user, mention, timestamp=str(tweet.date))

def scrape_tweets(limit=10):
    sample_tweets = [
        {"user": "alice", "mentions": ["bob", "carol"]},
        {"user": "bob", "mentions": ["dave"]},
        {"user": "carol", "mentions": ["eve", "frank"]},
        {"user": "dave", "mentions": []},
        {"user": "eve", "mentions": ["alice"]},
    ]

    for tweet in sample_tweets[:limit]:
        user = tweet["user"]
        mentions = tweet["mentions"]
        G.add_node(user)
        for mention in mentions:
            G.add_node(mention)
            G.add_edge(user, mention, timestamp="mock")

def show_graph_stats():
    print("🧠 Total nodes:", G.number_of_nodes())
    print("🔗 Total edges:", G.number_of_edges())
    print("📈 Sample edges:", list(G.edges(data=True))[:5])

if __name__ == "__main__":
    while True:
        print(" Scraping...")
        scrape_tweets(limit=5)
        show_graph_stats()
        print(" Current nodes:", list(G.nodes()))
        print(" Current edges:", list(G.edges()))

        draw_graph()
        plt.show(block=True)

        time.sleep(10)

