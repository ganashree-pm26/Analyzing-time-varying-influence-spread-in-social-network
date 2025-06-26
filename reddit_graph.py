import praw

# Reddit app credentials
reddit = praw.Reddit(
    client_id="b0Ppq9wrYTzYyLFNzZqjcw",
    client_secret="ybua2_kSmdQK_atcgCnbIyda0ixbfw",
    user_agent="RedditGraphStream by Ganashree (contact: ganashreepm@gmail.com)",
    username="Mundane-Campaign-791",
    password="Jyoreddit3&"
)

subreddit = reddit.subreddit("AskReddit")
print("Listening for new Reddit posts...")

for post in subreddit.stream.submissions(skip_existing=True):
    print(f"{post.title} - {post.score}")