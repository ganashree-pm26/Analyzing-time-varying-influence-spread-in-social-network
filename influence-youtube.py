from googleapiclient.discovery import build
import sys
import io

# Fix Unicode output for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# YouTube API setup
API_KEY = 'AIzaSyAlSgZ6Mess9MnELTEsLJTMRRn3u-plwu4'
youtube = build('youtube', 'v3', developerKey=API_KEY)

def search_trending_channels(keyword="AI", max_results=50):
    request = youtube.search().list(
        q=keyword,
        part="snippet",
        type="video",
        maxResults=max_results,
        order="viewCount"
    )
    response = request.execute()
    channels = {}

    for item in response['items']:
        channel_id = item['snippet']['channelId']
        channel_title = item['snippet']['channelTitle']
        channels[channel_id] = channel_title

    return channels

def get_channel_stats(channel_ids):
    stats = []

    for i in range(0, len(channel_ids), 50):
        batch_ids = list(channel_ids.keys())[i:i+50]

        response = youtube.channels().list(
            part="statistics",
            id=",".join(batch_ids)
        ).execute()

        for item in response["items"]:
            cid = item["id"]
            data = item["statistics"]
            title = channel_ids[cid]

            subs = int(data.get("subscriberCount", 0))
            views = int(data.get("viewCount", 0))
            videos = int(data.get("videoCount", 0))
            
            # Influence score (custom formula, can be improved)
            score = (subs * 0.5) + (views * 0.3) + (videos * 10)
            
            stats.append({
                "channel": title,
                "subscribers": subs,
                "views": views,
                "videos": videos,
                "score": score
            })

    return sorted(stats, key=lambda x: x["score"], reverse=True)[:15]

# Run the functions
if __name__ == "__main__":
    channels = search_trending_channels(keyword="AI", max_results=50)
    top_creators = get_channel_stats(channels)

    print("🔥 Top 15 Most Influential YouTube Creators on 'AI':\n")
    for i, creator in enumerate(top_creators, 1):
        print(f"{i}. {creator['channel']}")
        print(f"   📊 Subs: {creator['subscribers']}, Views: {creator['views']}, Videos: {creator['videos']}")
        print(f"   💡 Influence Score: {round(creator['score'], 2)}\n")
