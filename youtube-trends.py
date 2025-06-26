from googleapiclient.discovery import build
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

api_key = 'AIzaSyAlSgZ6Mess9MnELTEsLJTMRRn3u-plwu4'
youtube = build('youtube', 'v3', developerKey=api_key)

def search_youtube_trending(keyword, max_results=10):
    request = youtube.search().list(
        q=keyword,
        part="snippet",
        type="video",
        order="viewCount",
        maxResults=max_results
    )
    response = request.execute()

    results = []
    for item in response['items']:
        title = item['snippet']['title']
        channel = item['snippet']['channelTitle']
        video_id = item['id']['videoId']
        link = f"https://www.youtube.com/watch?v={video_id}"
        results.append((title, channel, link))

    return results

# Example usage
videos = search_youtube_trending("AI", max_results=15)
for title, channel, link in videos:
    print(f"{title} by {channel}\n {link}\n")
