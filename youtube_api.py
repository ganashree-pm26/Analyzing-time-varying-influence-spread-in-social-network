from flask import jsonify
from googleapiclient.discovery import build
import time

# Import configuration
from config import YOUTUBE_API_KEY, YOUTUBE_CACHE_DURATION

# Initialize YouTube API
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Cache for YouTube results
YOUTUBE_CACHE = {}

def get_cached_youtube_results(query, max_results=15):
    """Cache YouTube results to avoid hitting quota limits"""
    cache_key = f"{query}_{max_results}"
    current_time = time.time()
    
    # Check cache
    if cache_key in YOUTUBE_CACHE:
        cached_result = YOUTUBE_CACHE[cache_key]
        if current_time - cached_result['timestamp'] < YOUTUBE_CACHE_DURATION:
            return cached_result['data']
    
    try:
        # Make API request with minimal fields to save quota
        request = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results,
            order="viewCount",
            regionCode="IN",
            relevanceLanguage="hi,en",
            fields="items(id(videoId),snippet(title,channelTitle,description,thumbnails(high)))"
        )
        response = request.execute()

        videos = []
        for item in response['items']:
            snippet = item['snippet']
            video_data = {
                "title": snippet['title'],
                "channel": snippet['channelTitle'],
                "videoId": item['id']['videoId'],
                "description": snippet.get('description', ''),
                "thumbnail": snippet['thumbnails']['high']['url'],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            }
            videos.append(video_data)
        
        # Cache the results
        YOUTUBE_CACHE[cache_key] = {
            'timestamp': current_time,
            'data': videos
        }
        
        return videos
        
    except Exception as e:
        print(f"YouTube API error: {str(e)}")
        return get_sample_videos()

def get_sample_videos():
    """Return sample videos when API fails"""
    return [
        {
            "title": "Latest AI Developments in India",
            "channel": "Tech Burner",
            "videoId": "vr7UkUhF12g",
            "description": "Exploring the latest AI developments in India's tech ecosystem",
            "thumbnail": "https://i.ytimg.com/vi/vr7UkUhF12g/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=vr7UkUhF12g"
        },
        {
            "title": "AI Revolution in Indian Startups",
            "channel": "Varun Mayya",
            "videoId": "d8I5KpkR1V4",
            "description": "How Indian startups are leveraging AI technology",
            "thumbnail": "https://i.ytimg.com/vi/d8I5KpkR1V4/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=d8I5KpkR1V4"
        },
        {
            "title": "Machine Learning for Indian Developers",
            "channel": "Code With Harry",
            "videoId": "gfDE2a7MKjA",
            "description": "Complete tutorial on ML for Indian developers",
            "thumbnail": "https://i.ytimg.com/vi/gfDE2a7MKjA/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=gfDE2a7MKjA"
        }
    ]

def search_channels(keyword="AI", max_results=30):
    try:
        search = youtube.search().list(
            q=keyword,
            part="snippet",
            type="video",
            maxResults=max_results,
            order="viewCount"
        ).execute()
        return {i['snippet']['channelId']: i['snippet']['channelTitle'] for i in search['items']}
    except Exception as e:
        print(f"Error searching channels: {str(e)}")
        return {}

def get_channel_data(channel_ids):
    try:
        stats = []
        ids = list(channel_ids.keys())
        for i in range(0, len(ids), 50):
            response = youtube.channels().list(
                part="statistics",
                id=",".join(ids[i:i+50])
            ).execute()
            for ch in response["items"]:
                cid = ch["id"]
                s = ch["statistics"]
                subs = int(s.get("subscriberCount", 0))
                views = int(s.get("viewCount", 0))
                videos = int(s.get("videoCount", 0))
                score = (subs * 0.5) + (views * 0.3) + (videos * 10)
                stats.append({
                    "channel": channel_ids[cid],
                    "subscribers": subs,
                    "views": views,
                    "videos": videos,
                    "score": round(score, 2)
                })
        return sorted(stats, key=lambda x: x['score'], reverse=True)[:15]
    except Exception as e:
        print(f"Error getting channel data: {str(e)}")
        return [] 