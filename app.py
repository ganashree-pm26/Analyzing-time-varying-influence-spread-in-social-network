from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template, request  # type: ignore
from googleapiclient.discovery import build  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk import pos_tag  # type: ignore
from sklearn.mixture import GaussianMixture
from dateutil import parser  # type: ignore

import praw  # type: ignore
import networkx as nx  # type: ignore
import nltk  # type: ignore
import random
import time
import os
import json
import math
import numpy as np
import socket
import psutil
import threading
import queue
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
import io
from functools import lru_cache
from youtube_api import get_cached_youtube_results, get_sample_videos, search_channels, get_channel_data

# Global variables for traffic spike management
class TrafficSpikeManager:
    def __init__(self):
        self.servers = [
            {"id": "server-1", "location": "US-East", "ip": "192.168.1.10", "load": 0, "status": "active"},
            {"id": "server-2", "location": "US-West", "ip": "192.168.1.11", "load": 0, "status": "active"},
            {"id": "server-3", "location": "EU-Central", "ip": "192.168.1.12", "load": 0, "status": "active"},
            {"id": "server-4", "location": "Asia-Pacific", "ip": "192.168.1.13", "load": 0, "status": "active"},
            {"id": "server-5", "location": "US-Central", "ip": "192.168.1.14", "load": 0, "status": "active"}
        ]
        self.round_robin_index = 0
        self.traffic_history = deque(maxlen=100)
        self.spike_threshold = 80
        self.anycast_enabled = True
        
    def simulate_traffic_spike(self):
        """Simulate incoming traffic"""
        base_traffic = random.randint(10, 30)
        spike_chance = random.random()
        
        if spike_chance < 0.3:  # 30% chance of spike
            traffic = base_traffic + random.randint(50, 150)
        else:
            traffic = base_traffic
            
        timestamp = datetime.now()
        self.traffic_history.append({"timestamp": timestamp, "traffic": traffic})
        return traffic
    
    def round_robin_balance(self, requests):
        """Distribute requests using Round Robin algorithm"""
        active_servers = [s for s in self.servers if s["status"] == "active"]
        if not active_servers:
            return []
        
        distribution = []
        requests_per_server = requests // len(active_servers)
        remaining_requests = requests % len(active_servers)
        
        for i, server in enumerate(active_servers):
            allocated = requests_per_server
            if i < remaining_requests:
                allocated += 1
            
            server["load"] = min(100, server["load"] + allocated * 2)  # Simulate load increase
            distribution.append({
                "server_id": server["id"],
                "requests": allocated,
                "load": server["load"]
            })
        
        return distribution
    
    def anycast_route(self, client_location):
        """Simulate Anycast DNS routing to nearest server"""
        location_mapping = {
            "US": ["server-1", "server-2", "server-5"],
            "EU": ["server-3"],
            "ASIA": ["server-4"]
        }
        
        preferred_servers = location_mapping.get(client_location, ["server-1"])
        available_servers = [s for s in self.servers if s["id"] in preferred_servers and s["status"] == "active"]
        
        if not available_servers:
            available_servers = [s for s in self.servers if s["status"] == "active"]
        
        if available_servers:
            # Choose server with lowest load
            return min(available_servers, key=lambda x: x["load"])
        return None
    
    def detect_spike(self):
        """Detect if current traffic constitutes a spike"""
        if len(self.traffic_history) < 5:
            return False
        
        recent_traffic = [t["traffic"] for t in list(self.traffic_history)[-5:]]
        avg_traffic = sum(recent_traffic) / len(recent_traffic)
        return avg_traffic > self.spike_threshold
    
    def get_server_status(self):
        """Get current status of all servers"""
        return self.servers.copy()
    
    def reset_server_loads(self):
        """Reset server loads (simulate load decrease over time)"""
        for server in self.servers:
            server["load"] = max(0, server["load"] - random.randint(5, 15))

# Initialize traffic manager
traffic_manager = TrafficSpikeManager()



YOUTUBE_API_KEY = 'AIzaSyAlSgZ6Mess9MnELTEsLJTMRRn3u-plwu4'
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# --- Downloads ---
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

suspected_fake_reddit_users = set()
suspected_fake_youtube_users = set()
suspected_duplicate_titles = set()
flagged_users = set()
reddit_post_registry = {}


app = Flask(__name__)
G = nx.DiGraph()
influential_users = []

# --- Reddit API Setup ---
reddit = praw.Reddit(
    client_id="b0Ppq9wrYTzYyLFNzZqjcw",
    client_secret="ybua2_kSmdQK_atcgCnbIyda0ixbfw",
    user_agent="TrendGraphBatch by Ganashree"
)

# List of Indian subreddits to monitor
INDIAN_SUBREDDITS = [
    "india",
    "indianstartups",
    "IndianTechnology",
    "indiantech",
    "IndianGaming",
    "IndianProgramming",
    "IndianAcademia",
    "indianews",
    "IndiaSpeaks",
    "bangalore",
    "mumbai",
    "delhi",
    "IndianDevelopers",
    "IndianTeenagers",
    "IndianEngineer",
    "IndianArtificalIntelligence"
]

def is_indian_content(post):
    """Check if the content is related to India"""
    if not post or not post.title:
        return False
        
    # Check if post is from Indian subreddit
    if post.subreddit.display_name.lower() in [s.lower() for s in INDIAN_SUBREDDITS]:
        return True
        
    # Check title for Indian context
    indian_keywords = {
        'india', 'indian', 'bharat', 'bharatiya',
        'delhi', 'mumbai', 'bangalore', 'hyderabad', 'chennai',
        'pune', 'kolkata', 'ahmedabad', 'bengaluru',
        'rupee', 'rupees', 'inr', '₹'
    }
    
    title_words = set(post.title.lower().split())
    if any(keyword in title_words for keyword in indian_keywords):
        return True
        
    return False

# --- Filters ---
stop_words = set(stopwords.words("english"))
useless_words = stop_words.union({
    "get", "make", "thing", "good", "bad", "great", "today", "tomorrow", "going",
    "day", "time", "people", "new", "one", "still", "know", "much","years", "ice", "dog", "game","daughter" "really", "see",
    "many", "even", "feel", "want", "need", "just", "give", "take", "every", "back",
    "always","way","parts","part","jet", "engine","man","guys","til", "case", "cuts", "return", "thought", "done", "better", "worst", "best", "someone", "anyone", "help", "couch"
})

# --- Global Structures ---
trend_graphs = defaultdict(nx.Graph)
trend_users = defaultdict(set)
word_counter = Counter()

# --- Clean & Extract Words ---
def process_title(title):
    words = word_tokenize(title.lower())
    filtered = [w for w in words if w.isalpha() and w not in useless_words]
    tagged = pos_tag(filtered, tagset='universal')
    return [w for w, tag in tagged if tag == "NOUN" and len(w) > 2]

def refresh_data_periodically():
    while True:
        collect_posts()
        print("[Refresh] Data collected again.")
        time.sleep(3600)  # Refresh every hour

def influence_spike_detected(username):
    combined = nx.compose_all(trend_graphs.values())
    now = datetime.utcnow()
    spike_count = 0

    for u, v, data in combined.edges(data=True):
        if u == username:
            try:
                edge_time = datetime.strptime(data.get("time", "00:00:00"), "%H:%M:%S")
                elapsed_minutes = (now - edge_time).seconds / 60
                if elapsed_minutes <= 5:
                    spike_count += 1
            except:
                continue

    return spike_count >= 3  # at least 3 spikes in 5 minutes # Threshold: 3+ spikes in 5 min


def uses_https(username):
    combined = nx.compose_all(trend_graphs.values())
    for u, v, data in combined.edges(data=True):
        if u == username or v == username:
            platform = data.get("platform", "").lower()
            if "https" in platform:
                return True
    return False


def is_duplicate_content(username):
    combined = nx.compose_all(trend_graphs.values())
    seen_edges = set()
    duplicate_count = 0

    for u, v, data in combined.edges(data=True):
        if u == username:
            key = (v, data.get("time", ""))
            if key in seen_edges:
                duplicate_count += 1
            else:
                seen_edges.add(key)

    return duplicate_count >= 2  # 2 or more repeats


def build_routing_graph():
    routing_graph = nx.DiGraph()
    for trend, G in trend_graphs.items():
        for u, v, data in G.edges(data=True):
            weight = 1 - random.uniform(0.1, 0.9)  # Replace with real probability or time
            routing_graph.add_edge(u, v, weight=weight)
    return routing_graph

def calculate_fake_score(user, reasons):
    score = 0
    for reason in reasons:
        if "sudden influence" in reason.lower():
            score += 5
        elif "https" in reason.lower():
            score += 2
        elif "duplicate" in reason.lower():
            score += 3
        else:
            score += 1
    return min(score, 10)



@app.route('/api/fake-users')
def fake_users_api():
    return jsonify({
        "reddit": list(suspected_fake_reddit_users),
        "duplicates": list(suspected_duplicate_titles)
    })

@app.route('/api/flagged-fake-users')
def flagged_fake_users():
    return jsonify({'users': list(flagged_users)})

@app.route('/api/fake-score', methods=['GET'])
def get_fake_score():
    user = request.args.get('user')
    reasons = request.args.getlist('reason')
    score = calculate_fake_score(user, reasons)
    return jsonify({'user': user, 'fake_score': score})

@app.route('/detect-fake-news')
def show_detect_fake_news():
    return render_template('detect_fake_news.html')

@app.route('/api/influence-path/<source>/<target>')
def influence_path(source, target):
    # Combine all trend graphs to get the complete network
    G = nx.compose_all(trend_graphs.values())
    
    try:
        # Find all simple paths between source and target
        all_paths = list(nx.all_simple_paths(G, source, target, cutoff=5))
        
        if not all_paths:
            return jsonify({'error': 'No path found between users'}), 404
            
        # For each path, calculate its total weight
        path_weights = []
        for path in all_paths:
            total_weight = 0
            edges = list(zip(path[:-1], path[1:]))
            for u, v in edges:
                # Get edge data
                edge_data = G[u][v]
                weight = edge_data.get('weight', 0)
                timestamp = edge_data.get('time', '00:00:00')
                
                # Calculate time-based weight
                try:
                    edge_time = datetime.strptime(timestamp, "%H:%M:%S")
                    now = datetime.utcnow()
                    hours_ago = (now - edge_time).seconds / 3600
                    
                    # Less aggressive decay for 48-hour window
                    time_factor = 1.0 / (1.0 + (hours_ago / 24))  # Decay over 24-hour periods
                    adjusted_weight = weight * time_factor
                except:
                    adjusted_weight = weight
                    
                total_weight += adjusted_weight
            
            path_weights.append((path, total_weight, edges))
        
        # Sort paths by weight and get the best one
        best_path = max(path_weights, key=lambda x: x[1])
        
        return jsonify({
            'path': best_path[0],
            'edges': best_path[2],
            'weight': round(best_path[1], 3),
            'total_paths': len(all_paths)
        })
        
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return jsonify({'error': 'No path found or user not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check-fake-score', methods=['POST'])
def check_fake_score():
    username = request.json.get('username')
    
    # Real-time calculation
    reasons = []
    score = 0
    
    # For any user (flagged or not), check all conditions
    # 1. Check for sudden influence spike
    if influence_spike_detected(username):
        reasons.append("⚠ Sudden influence spike in under 5 minutes")
        score += 5
        suspected_fake_reddit_users.add(username)
        
    # 2. Check for unverified source
    if uses_https(username):
        reasons.append("🔒 Source does not use HTTPS")
        score += 2
        suspected_fake_reddit_users.add(username)
        
    # 3. Check for duplicate content
    if is_duplicate_content(username):
        reasons.append("📄 Duplicate content from another user")
        score += 3
        suspected_fake_reddit_users.add(username)
    
    # If user is already flagged but no conditions were detected, add default reasons
    if username in suspected_fake_reddit_users and not reasons:
        reasons = [
            "⚠ Sudden influence spike in under 5 minutes",
            "🔒 Source does not use HTTPS",
            "📄 Duplicate content from another user"
        ]
        score = 10  # Maximum score for flagged users
    
    # If there are manually selected reasons, add them too
    selected_reasons = request.json.get('reasons', [])
    if selected_reasons:
        if "Sudden influence spike" in str(selected_reasons):
            if "⚠ Sudden influence spike in under 5 minutes" not in reasons:
                reasons.append("⚠ Sudden influence spike in under 5 minutes")
                score += 5
        if "Source does not use HTTPS" in str(selected_reasons):
            if "🔒 Source does not use HTTPS" not in reasons:
                reasons.append("🔒 Source does not use HTTPS")
                score += 2
        if "Duplicate content" in str(selected_reasons):
            if "📄 Duplicate content from another user" not in reasons:
                reasons.append("📄 Duplicate content from another user")
                score += 3
    
    return jsonify({
        'score': min(score, 10),
        'max': 10,
        'username': username,
        'reasons': reasons
    })

# Simplified TrafficFlowAnalyzer class
class TrafficFlowAnalyzer:
    def __init__(self):
        self.traffic_data = []
        self.is_collecting = False
        self.anomalies = []
        self.hmm_model = None  # Initialize HMM model as None
        
    def collect_network_stats(self):
        """Collect real-time network statistics"""
        stats = psutil.net_io_counters(pernic=True)
        connections = len(psutil.net_connections())
        
        # Aggregate network data
        total_bytes_sent = sum([stat.bytes_sent for stat in stats.values()])
        total_bytes_recv = sum([stat.bytes_recv for stat in stats.values()])
        total_packets_sent = sum([stat.packets_sent for stat in stats.values()])
        total_packets_recv = sum([stat.packets_recv for stat in stats.values()])
        
        return {
            'timestamp': datetime.now(),
            'bytes_sent': total_bytes_sent,
            'bytes_recv': total_bytes_recv,
            'packets_sent': total_packets_sent,
            'packets_recv': total_packets_recv,
            'connections': connections
        }
    
    def detect_anomalies(self, current_data):
        """Simple anomaly detection based on standard deviation"""
        if len(self.traffic_data) < 10:
            return False, 0.0
            
        # Calculate mean and standard deviation of recent data
        recent_data = self.traffic_data[-100:]
        bytes_sent = [d['bytes_sent'] for d in recent_data]
        mean_sent = np.mean(bytes_sent)
        std_sent = np.std(bytes_sent)
        
        # Check if current data is an anomaly (more than 3 standard deviations from mean)
        z_score = (current_data['bytes_sent'] - mean_sent) / (std_sent + 1e-8)
        is_anomaly = abs(z_score) > 3
        
        if is_anomaly:
            self.anomalies.append({
                'timestamp': current_data['timestamp'],
                'likelihood': -abs(z_score),
                'data': current_data
            })
        
        return is_anomaly, -abs(z_score)

# Global traffic analyzer
traffic_analyzer = TrafficFlowAnalyzer()

def collect_traffic_data():
    """Background thread to collect traffic data"""
    while traffic_analyzer.is_collecting:
        try:
            data = traffic_analyzer.collect_network_stats()
            traffic_analyzer.traffic_data.append(data)
            
            # Keep only last 1000 data points
            if len(traffic_analyzer.traffic_data) > 1000:
                traffic_analyzer.traffic_data = traffic_analyzer.traffic_data[-1000:]
            
            # Detect anomalies
            if len(traffic_analyzer.traffic_data) > 50:
                is_anomaly, likelihood = traffic_analyzer.detect_anomalies(data)
                if is_anomaly:
                    print(f"Anomaly detected at {data['timestamp']}: likelihood = {likelihood}")
            
            time.sleep(2)  # Collect data every 2 seconds
            
        except Exception as e:
            print(f"Error collecting traffic data: {e}")
            time.sleep(5)

@app.route('/traffic-analysis')
def traffic_analysis_page():
    return render_template('traffic_analysis.html')

@app.route('/api/start-traffic-monitoring')
def start_traffic_monitoring():
    if not traffic_analyzer.is_collecting:
        traffic_analyzer.is_collecting = True
        threading.Thread(target=collect_traffic_data, daemon=True).start()
        return jsonify({'status': 'started', 'message': 'Traffic monitoring started'})
    return jsonify({'status': 'already_running', 'message': 'Traffic monitoring already running'})

@app.route('/api/stop-traffic-monitoring')
def stop_traffic_monitoring():
    traffic_analyzer.is_collecting = False
    return jsonify({'status': 'stopped', 'message': 'Traffic monitoring stopped'})

@app.route('/api/traffic-data')
def get_traffic_data():
    # Get last 50 data points for visualization
    recent_data = traffic_analyzer.traffic_data[-50:] if traffic_analyzer.traffic_data else []
    
    formatted_data = []
    for data in recent_data:
        formatted_data.append({
            'timestamp': data['timestamp'].strftime('%H:%M:%S'),
            'bytes_sent': data['bytes_sent'],
            'bytes_recv': data['bytes_recv'],
            'packets_sent': data['packets_sent'],
            'packets_recv': data['packets_recv'],
            'connections': data['connections']
        })
    
    return jsonify({
        'data': formatted_data,
        'total_points': len(traffic_analyzer.traffic_data),
        'model_trained': False  # Simplified to always return False since we're not using HMM
    })

@app.route('/api/traffic-anomalies')
def get_traffic_anomalies():
    # Get recent anomalies
    recent_anomalies = traffic_analyzer.anomalies[-20:] if traffic_analyzer.anomalies else []
    
    formatted_anomalies = []
    for anomaly in recent_anomalies:
        formatted_anomalies.append({
            'timestamp': anomaly['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'likelihood': round(anomaly['likelihood'], 4),
            'bytes_sent': anomaly['data']['bytes_sent'],
            'bytes_recv': anomaly['data']['bytes_recv'],
            'connections': anomaly['data']['connections']
        })
    
    return jsonify({
        'anomalies': formatted_anomalies,
        'total_anomalies': len(traffic_analyzer.anomalies)
    })

@app.route('/api/traffic-stats')
def get_traffic_stats():
    if not traffic_analyzer.traffic_data:
        return jsonify({'error': 'No data available'})
    
    recent_data = traffic_analyzer.traffic_data[-100:] if len(traffic_analyzer.traffic_data) >= 100 else traffic_analyzer.traffic_data
    
    avg_bytes_sent = np.mean([d['bytes_sent'] for d in recent_data])
    avg_bytes_recv = np.mean([d['bytes_recv'] for d in recent_data])
    avg_connections = np.mean([d['connections'] for d in recent_data])
    
    return jsonify({
        'avg_bytes_sent': round(avg_bytes_sent, 2),
        'avg_bytes_recv': round(avg_bytes_recv, 2),
        'avg_connections': round(avg_connections, 2),
        'anomaly_rate': len(traffic_analyzer.anomalies) / len(traffic_analyzer.traffic_data) * 100 if traffic_analyzer.traffic_data else 0
    })

@app.route("/influence-path")
def influence_path_page():
    return render_template("influence_path.html")

@app.route('/api/full-graph')
def full_graph():
    # Combine all trend graphs to get the complete network
    combined = nx.compose_all(trend_graphs.values())
    
    # Calculate node sizes based on degree centrality
    centrality = nx.degree_centrality(combined)
    max_centrality = max(centrality.values()) if centrality else 1
    
    nodes = []
    for node in combined.nodes():
        # Scale node size between 20 and 60 based on centrality
        size = 20 + (centrality[node] / max_centrality * 40)
        nodes.append({
            "data": {
                "id": node,
                "size": size,
                "connections": combined.degree(node)
            }
        })
    
    edges = []
    for u, v, data in combined.edges(data=True):
        # Get edge weight and normalize it
        weight = data.get('weight', 0.1)
        # Scale edge width between 1 and 8
        width = 1 + (weight * 7)
        
        edges.append({
            "data": {
                "source": u,
                "target": v,
                "weight": round(width, 2),
                "original_weight": round(weight, 3)
            }
        })
    
    return jsonify({'nodes': nodes, 'edges': edges})

@app.route('/api/find-path')
def find_path():
    source = request.args.get('source')
    target = request.args.get('target')
    G = nx.compose_all(trend_graphs.values())

    try:
        path = nx.dijkstra_path(G, source=source, target=target, weight='weight')
        total_weight = round(nx.dijkstra_path_length(G, source=source, target=target, weight='weight'), 2)
        return jsonify({'path': path, 'weight': total_weight})
    except Exception:
        return jsonify({'path': [], 'weight': 0})

from flask import request # type: ignore

@app.route('/api/fake-detections')
def fake_detection_api():
    suspicious_users = []
    now = datetime.utcnow()

    for trend, G in trend_graphs.items():
        for node in G.nodes(data=True):
            username = node[0]
            user_time = node[1].get('time')

            if not user_time:
                continue

            try:
                t_user = datetime.strptime(user_time, "%H:%M:%S")
                minutes_ago = (now - t_user).total_seconds() / 60
                if minutes_ago <= 5:  # Recent spike
                    influence_count = G.degree(username)
                    if influence_count >= 15:  # High spread in a short time
                        suspicious_users.append({
                            "user": username,
                            "reason": "⚠ Sudden influence spike in under 5 minutes"
                        })
            except Exception:
                continue

    # Example unverified pattern (no HTTPS)
    for trend in trend_graphs:
        for u, v, data in trend_graphs[trend].edges(data=True):
            if 'http' in u.lower() and 'https' not in u.lower():
                suspicious_users.append({
                    "user": u,
                    "reason": "🔒 Source does not use HTTPS"
                })

    # Deduplicate users
    seen = set()
    cleaned = []
    for entry in suspicious_users:
        if entry["user"] not in seen:
            cleaned.append(entry)
            seen.add(entry["user"])

    return jsonify(cleaned)

@app.route('/detect-fake-news')
def detect_fake_news():
    return render_template('fake_detection.html')

@app.route('/routing')
def routing_page():
    return render_template('routing_path.html')

@app.route('/dijkstra-path')
def dijkstra_path_page():
    return render_template('dijkstra_path.html')

@app.route('/api/dijkstra-network')
def dijkstra_network():
    combined = nx.compose_all(trend_graphs.values())
    elements = []

    for node in combined.nodes:
        elements.append({'data': {'id': node}})
        
    for u, v, d in combined.edges(data=True):
        elements.append({
            'data': {
                'source': u,
                'target': v,
                'weight': d.get('weight', 1),
                'platform': d.get('platform', 'Unknown'),
                'timestamp': d.get('timestamp', 'N/A'),
                'location': d.get('location', 'Unknown')
            }
        })

    return jsonify({'elements': elements})


@app.route('/api/find-influence-path')
def find_influence_path():
    source = request.args.get('source')
    target = request.args.get('target')

    combined = nx.compose_all(trend_graphs.values())

    try:
        path = nx.dijkstra_path(combined, source=source, target=target, weight='weight')
        return jsonify({'path': path})
    except Exception as e:
        return jsonify({'error': str(e), 'path': None})


def independent_cascade(G, seeds, p=0.1, iterations=100):
    spread = 0
    for _ in range(iterations):
        active = set(seeds)
        new_active = set(seeds)

        while new_active:
            next_active = set()
            for node in new_active:
                for neighbor in G.neighbors(node):
                    if neighbor not in active:
                        # Use edge weight if available, else default to p
                        weight = G[node][neighbor].get('weight', p)
                        if random.random() <= weight:
                            next_active.add(neighbor)

            active.update(next_active)
            new_active = next_active

        spread += len(active)
    return spread / iterations

def celf(G, k=5, p=0.1):
    node_list = list(G.nodes)
    marginal_gain = []

    # Initial gain for each node
    for node in node_list:
        mg = independent_cascade(G, [node], p)
        marginal_gain.append((mg, node))

    # Sort by gain
    marginal_gain.sort(reverse=True)
    selected = []
    spread = 0
    visited = [0] * len(G.nodes)

    for _ in range(k):
        while True:
            mg, node = marginal_gain[0]
            if visited[node_list.index(node)] == len(selected):
                selected.append(node)
                spread += mg
                break
            new_mg = independent_cascade(G, selected + [node], p) - spread
            marginal_gain[0] = (new_mg, node)
            visited[node_list.index(node)] = len(selected)
            marginal_gain.sort(reverse=True)

    return selected

def build_youtube_graph(channels):
    G = nx.DiGraph()
    G.add_nodes_from(channels)

    for _ in range(len(channels) * 2):  # simulate possible influence
        a, b = random.sample(channels, 2)
        G.add_edge(a, b)  # a influences b

    return G

def fetch_trending_channels(region="IN", max_results=25):
    trending_videos = youtube.videos().list(
        part="snippet",
        chart="mostPopular",
        maxResults=max_results,
        regionCode="IN",  # Explicitly set to India
        videoCategoryId="28"  # Science & Technology category
    ).execute()

    channels = set()
    channel_map = {}

    for item in trending_videos['items']:
        title = item['snippet']['title']
        channel = item['snippet']['channelTitle']
        channel_id = item['snippet']['channelId']
        channels.add(channel_id)
        channel_map[channel_id] = channel

    return list(channels), channel_map

def is_suspect_post(post):
    reasons = []
    now = datetime.utcnow()
    post_time = datetime.utcfromtimestamp(post.created_utc)

    # Only check posts from the last 48 hours
    if (now - post_time).total_seconds() > 172800:  # 48 hours in seconds
        return []

    # Check for sudden influence spike
    if (now - post_time).seconds <= 300:  # 5 minutes
        # Check for high engagement in short time
        if hasattr(post, 'score') and post.score > 50:  # High score in 5 minutes is suspicious
            reasons.append("sudden_spike")
        
        # Also check comment velocity
        if hasattr(post, 'num_comments') and post.num_comments > 20:  # Many comments in 5 minutes
            reasons.append("sudden_spike")

    # Check for HTTPS
    if hasattr(post, 'url') and post.url:
        # Only flag if it's a content link (not a self post) and doesn't use HTTPS
        if not post.is_self and not post.url.startswith("https"):
            reasons.append("non_https")

    # Check for duplicate content
    title = post.title.strip().lower()
    if title in reddit_post_registry:
        # Only flag if the same content appears multiple times from different users
        original_author = reddit_post_registry[title]
        if original_author != post.author.name:
            reasons.append("duplicate_content")
            suspected_duplicate_titles.add(title)
    else:
        reddit_post_registry[title] = post.author.name

    # Additional checks for suspicious behavior
    if hasattr(post, 'author'):
        # Check account age
        account_age = (now - datetime.utcfromtimestamp(post.author.created_utc)).days
        if account_age < 30:  # Account less than 30 days old
            reasons.append("new_account")
        
        # Check karma ratio
        if hasattr(post.author, 'comment_karma') and hasattr(post.author, 'link_karma'):
            total_karma = post.author.comment_karma + post.author.link_karma
            if total_karma < 100 and account_age > 30:  # Low karma despite age
                reasons.append("low_engagement")

    return list(set(reasons))  # Remove any duplicates


# --- Collect Data (Top 4 hours)
def collect_posts():
    print("[INFO] Collecting top posts from Indian subreddits...")
    now = datetime.utcnow()
    cutoff = now - timedelta(hours=48)
    post_count = 0
    flagged_count = 0
    
    # Create a multireddit of Indian subreddits
    subreddits = '+'.join(INDIAN_SUBREDDITS)
    
    # Collect posts from different time periods to ensure coverage
    time_filters = ["hour", "day", "week"]
    posts_seen = set()
    
    for time_filter in time_filters:
        for post in reddit.subreddit(subreddits).top(time_filter=time_filter, limit=500):
            # Skip if post is too old
            if post.created_utc < cutoff.timestamp():
                continue
            
            # Skip if we've seen this post already
            if post.id in posts_seen:
                continue
                
            posts_seen.add(post.id)
            
            if not post.title or not post.author:
                continue
                
            # Only process Indian content
            if not is_indian_content(post):
                continue

            author = post.author.name
            words = process_title(post.title)
            timestamp = datetime.utcfromtimestamp(post.created_utc).strftime("%H:%M:%S")
            
            # Check for suspicious behavior
            reasons = is_suspect_post(post)
            if reasons:
                # Add to suspected list if ANY criteria is met (for monitoring)
                if len(reasons) >= 1:
                    suspected_fake_reddit_users.add(author)
                    flagged_count += 1
                
                # Add to flagged list if MULTIPLE criteria are met
                if len(reasons) >= 2:
                    flagged_users.add(author)

            # Build trend graph
            for word in words:
                word_counter[word] += 1
                G = trend_graphs[word]
                trend_users[word].add(author)

                G.add_node(author, time=timestamp)
                for other_user in trend_users[word]:
                    if other_user != author:
                        now = datetime.utcnow()
                        interaction_time = datetime.strptime(timestamp, "%H:%M:%S")
                        minutes_ago = (now - interaction_time).seconds / 60

                        # Influence decays exponentially with time
                        decay = round(1 / (1 + minutes_ago), 4)

                        edge_attrs = {
                            'weight': decay,
                            'platform': 'Reddit',
                            'time': timestamp,
                            'location': 'IN'
                        }
                           
                        if G.has_edge(author, other_user):
                            existing = G[author][other_user]['weight']
                            G[author][other_user]['weight'] = max(existing, decay)
                            G[author][other_user].update(edge_attrs)
                        else:
                            G.add_edge(author, other_user, **edge_attrs)
            
            post_count += 1

    print(f"[INFO] Finished collecting {post_count} posts from Indian subreddits.")
    print(f"[INFO] Flagged {flagged_count} suspicious users out of {post_count} posts.")
    print(f"[INFO] {len(flagged_users)} users met multiple criteria for fake detection.")

threading.Thread(target=refresh_data_periodically, daemon=True).start()

def build_routing_graph():
    routing_graph = nx.DiGraph()
    for trend, G in trend_graphs.items():
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)  # use actual time delay if available
            routing_graph.add_edge(u, v, weight=weight)
    return routing_graph


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/view-graph-trends')
def view_graph_trends():
    # Removed collect_posts() call to prevent blocking the server
    return render_template('network_trends.html')

@app.route('/taic-influencers')
def taic_influencers_page():
    return render_template('taic_influencers.html')

@app.route('/api/taic-influencers')
def taic_influencers_api():
    combined_graph = nx.compose_all(trend_graphs.values())
    top_users = celf(combined_graph, k=10)  # Now uses time-aware influence
    return jsonify({'influential_users': top_users})


# Add cache for YouTube results
YOUTUBE_CACHE = {}
CACHE_DURATION = 300  # 5 minutes

@lru_cache(maxsize=32)
def get_cached_youtube_results(query, max_results=15):
    """Cache YouTube results to avoid hitting quota limits"""
    cache_key = f"{query}_{max_results}"
    current_time = time.time()
    
    # Check cache
    if cache_key in YOUTUBE_CACHE:
        cached_result = YOUTUBE_CACHE[cache_key]
        if current_time - cached_result['timestamp'] < CACHE_DURATION:
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

def get_trending_ai_videos():
    """Return trending AI-related videos on YouTube from April and May 2025"""
    return [
        {
            "title": "AI TECH PULSE: Top Artificial Intelligence Developments of April 2025",
            "channel": "EconoEthos",
            "videoId": "xYBJxgUwY04",
            "description": "A comprehensive overview of the most significant AI developments in April 2025.",
            "thumbnail": "https://i.ytimg.com/vi/xYBJxgUwY04/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=xYBJxgUwY04"
        },
        {
            "title": "23rd May 2025 | Viral AI Videos You'll Watch on Repeat!",
            "channel": "Year Countdown Live",
            "videoId": "3cuS98_QgMM",
            "description": "A compilation of the most viral and trending AI-generated videos of the year.",
            "thumbnail": "https://i.ytimg.com/vi/3cuS98_QgMM/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=3cuS98_QgMM"
        },
        {
            "title": "5 AI Trends Right Now! (April 2025)",
            "channel": "ElevateWithAIwork",
            "videoId": "nzkNR52Jozw",
            "description": "An overview of the top 5 AI trends shaping the future of technology.",
            "thumbnail": "https://i.ytimg.com/vi/nzkNR52Jozw/hqdefault.jpg",
            "url": "https://www.youtube.com/shorts/nzkNR52Jozw"
        },
        {
            "title": "Ai Video #yt #trending #ai 24 May 2025",
            "channel": "Rakesh Highway Diary",
            "videoId": "JhSPfS3MhRg",
            "description": "A green screen reaction video featuring AI-generated content.",
            "thumbnail": "https://i.ytimg.com/vi/JhSPfS3MhRg/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=JhSPfS3MhRg"
        },
        {
            "title": "Jeff Dean's talk at ETH Zurich in April 2025 on important trends in AI",
            "channel": "Cong Xu",
            "videoId": "q6pAWOG_10k",
            "description": "Jeff Dean discusses the evolution and future of AI in this insightful talk.",
            "thumbnail": "https://i.ytimg.com/vi/q6pAWOG_10k/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=q6pAWOG_10k"
        },
        {
            "title": "#yt #trending #ai #short 16 May 2025",
            "channel": "Raa.React@",
            "videoId": "HuNln8CrMrg",
            "description": "An AI-generated video showcasing trending content.",
            "thumbnail": "https://i.ytimg.com/vi/HuNln8CrMrg/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=HuNln8CrMrg"
        },
        {
            "title": "Trending Topics Of May2025!! #shorts #trending #technology #2025",
            "channel": "Next!",
            "videoId": "ENm9F5E2kt8",
            "description": "A quick dive into the hottest trending topics of May 2025, including AI breakthroughs.",
            "thumbnail": "https://i.ytimg.com/vi/ENm9F5E2kt8/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=ENm9F5E2kt8"
        },
        {
            "title": "Trends in AI - April 2025 | Live from Hannover Messe!",
            "channel": "Zeta Alpha",
            "videoId": "fbHM8LmA8ps",
            "description": "Live coverage of AI trends from the world's leading industrial tech trade fair.",
            "thumbnail": "https://i.ytimg.com/vi/fbHM8LmA8ps/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=fbHM8LmA8ps"
        },
        {
            "title": "Top 11 AI Trends Defining 2025",
            "channel": "AI Explained",
            "videoId": "xoknlPcv2dA",
            "description": "An in-depth look at the top 11 AI trends set to define the year 2025.",
            "thumbnail": "https://i.ytimg.com/vi/xoknlPcv2dA/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=xoknlPcv2dA"
        },
        {
            "title": "The BEST AI Video Trends of 2025 Revealed!",
            "channel": "AI Trends",
            "videoId": "fkgshV88jSY",
            "description": "A sneak peek into the future of AI-powered videos and what to expect in 2025.",
            "thumbnail": "https://i.ytimg.com/vi/fkgshV88jSY/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=fkgshV88jSY"
        },
        {
            "title": "AI Trends and Predictions for 2025 | AI Rising",
            "channel": "Mint",
            "videoId": "QHQepJjmFxg",
            "description": "A comprehensive discussion on AI trends and predictions for 2025.",
            "thumbnail": "https://i.ytimg.com/vi/QHQepJjmFxg/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=QHQepJjmFxg"
        },
        {
            "title": "Top 10 AI Trends In 2025 Everyone Must Be Ready For Now",
            "channel": "Bernard Marr",
            "videoId": "_GxJjyKVzko",
            "description": "Discover the top 10 AI trends shaping 2025, from sustainable AI to autonomous agents.",
            "thumbnail": "https://i.ytimg.com/vi/_GxJjyKVzko/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=_GxJjyKVzko"
        },
        {
            "title": "AI Trends for 2025",
            "channel": "IBM Technology",
            "videoId": "5zuF4Ys1eAw",
            "description": "An exploration of the importance of AI agents and assistants in 2025.",
            "thumbnail": "https://i.ytimg.com/vi/5zuF4Ys1eAw/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=5zuF4Ys1eAw"
        },
        {
            "title": "AI Trends of 2025: What's Trending in AI in April?",
            "channel": "Envato Elements",
            "videoId": "AI_Trends_April_2025",
            "description": "A look at the super-powered generative AI tools trending in April 2025.",
            "thumbnail": "https://elements.envato.com/learn/ai-trends",
            "url": "https://elements.envato.com/learn/ai-trends"
        },
        {
            "title": "Top 8 AI Video Trends to Watch Out for in 2025",
            "channel": "Puppetry",
            "videoId": "Top_8_AI_Video_Trends_2025",
            "description": "Explore the top AI video trends revolutionizing content creation in 2025.",
            "thumbnail": "https://www.puppetry.com/posts/top-8-ai-video-trends-to-watch-out-for-in-2025",
            "url": "https://www.puppetry.com/posts/top-8-ai-video-trends-to-watch-out-for-in-2025"
        }
    ]

@app.route('/api/youtube-trending')
def youtube_trending_api():
    try:
        videos = get_cached_youtube_results("Indian AI technology")
        return jsonify(videos)
    except Exception as e:
        print(f"Error in youtube_trending_api: {str(e)}")
        return jsonify(get_sample_videos())

@app.route('/youtube-trending')
def youtube_trending_page():
    return render_template('youtube_trending.html')

@app.route('/api/youtube-influencers')
def youtube_influencers_api():
    channels = search_channels()
    top = get_channel_data(channels)
    return jsonify(top)

@app.route('/youtube-influences')
def show_youtube_influences():
    # Fetch channels and build graph
    channels, channel_map = fetch_trending_channels()
    G = build_youtube_graph(channels)

    # Run CELF algorithm to find top influencers
    top_ids = celf(G, k=10)
    top_names = [channel_map.get(ch_id, ch_id) for ch_id in top_ids]

    return render_template('youtube_dynamic.html', influencers=top_names)

@app.route('/youtube-dynamic')
def youtube_dynamic():
    return render_template('youtube_dynamic.html')


@app.route('/youtube-influencers')
def youtube_influencers_page():
    return render_template('youtube_influencers.html')

@app.route('/trends')
def trends():
    collect_posts()
    return jsonify({'trends': [w for w, _ in word_counter.most_common(10)]})

@app.route('/graph/<trend>')
def graph(trend):
    G = trend_graphs.get(trend, nx.Graph())
    nodes = list(G.nodes)
    edges = [{"source": u, "target": v} for u, v in G.edges]
    return jsonify({'nodes': nodes, 'edges': edges})

@app.route('/trending-topics')
def trending_topics_page():
    return render_template('trending_topics.html')

@app.route('/api/trend-networks')
def trend_networks():
    top_trends = [w for w, _ in word_counter.most_common(15)]
    trend_graph_data = {}

    for trend in top_trends:
        G = trend_graphs.get(trend, nx.Graph())
        nodes = [{"id": node} for node in G.nodes]
        edges = [{"source": u, "target": v} for u, v in G.edges()]
        trend_graph_data[trend] = {"nodes": nodes, "links": edges}

    return jsonify(trend_graph_data)

@app.route('/influential-users')
def influential_users_page():
    return render_template("influential_users.html")


@app.route('/api/influential-users')
def influential_users_api():
    combined_graph = nx.compose_all(trend_graphs.values())
    top_users = celf(combined_graph, k=10)
    return jsonify({'influential_users': top_users})

@app.route('/api/trend-graph')
def trend_graph_api():
    timestamps = []
    trends_data = defaultdict(lambda: [0]*10)  # Simulated 10 time points

    # Simulate or replace with actual historical counts if available
    for i in range(10):
        time_label = (datetime.utcnow() - timedelta(minutes=10 * (0.9 - i))).strftime('%H:%M')
        timestamps.append(time_label)
        for word, _ in word_counter.most_common(5):
            trends_data[word][i] = random.randint(3, 20)

    trends_json = [{
        'name': topic,
        'values': trends_data[topic],
        'color': f'#{random.randint(0, 0xFFFFFF):06x}'
    } for topic in trends_data]

    return jsonify({'timestamps': timestamps, 'trends': trends_json})

def get_recent_youtube_data(hours=48):
    try:
        now = datetime.utcnow()
        published_after = (now - timedelta(hours=hours)).isoformat() + 'Z'
        
        # Search queries focused on Indian tech content
        search_queries = [
            'Indian AI technology',
            'Indian tech startups',
            'Indian machine learning',
            'Indian data science',
            'Indian tech news'
        ]
        channels_data = {}
        
        for query in search_queries:
            request = youtube.search().list(
                q=query,
                part='snippet',
                type='video',
                maxResults=15,
                order='viewCount',
                regionCode='IN',  # Restrict to Indian content
                relevanceLanguage='hi,en',  # Hindi and English content
                publishedAfter=published_after
            )
            response = request.execute()
            
            for item in response['items']:
                channel_id = item['snippet']['channelId']
                channel_title = item['snippet']['channelTitle']
                
                if channel_id not in channels_data:
                    # Get channel statistics
                    channel_stats = youtube.channels().list(
                        part='statistics',
                        id=channel_id
                    ).execute()
                    
                    if channel_stats['items']:
                        stats = channel_stats['items'][0]['statistics']
                        subscriber_count = int(stats.get('subscriberCount', 0))
                        view_count = int(stats.get('viewCount', 0))
                        video_count = int(stats.get('videoCount', 0))
                        
                        # Calculate metrics
                        influence = min(1.0, (subscriber_count * 0.4 + view_count * 0.4 + video_count * 0.2) / 10**8)
                        subscriber_factor = min(1.0, subscriber_count / 10**7)
                        engagement_rate = min(1.0, view_count / (subscriber_count * video_count) if subscriber_count * video_count > 0 else 0)
                        
                        channels_data[channel_id] = {
                            'title': channel_title,
                            'influence': influence,
                            'subscriber_factor': subscriber_factor,
                            'engagement_rate': engagement_rate,
                            'stats': {
                                'subscribers': subscriber_count,
                                'views': view_count,
                                'videos': video_count
                            }
                        }
        
        return channels_data
    except Exception as e:
        print(f"Error fetching YouTube data: {str(e)}")
        return {}

@app.route('/api/youtube-network')
def youtube_network():
    try:
        # Indian tech/AI focused channels
        channels = [
            "Varun Mayya",
            "Tech Burner",
            "Technical Guruji",
            "Shashank Udupa",
            "Ankur Warikoo",
            "Tanay Pratap",
            "Code With Harry",
            "Apna College",
            "Love Babbar",
            "Kunal Kushwaha",
            "Telusko",
            "Krish Naik",
            "Indian AI Production",
            "Analytics India",
            "The Indian Dev",
            "GeeksforGeeks",
            "Indian Pythonista",
            "Indian Tech News",
            "DataFlair",
            "Indian Coder",
            "Tech With Tim India",
            "CodeWithChai",
            "IndianProgrammer",
            "AI India Today",
            "Indian Tech Solutions",
            "Bharat AI",
            "Code Keen India",
            "Indian Developer Hub",
            "Tech Dost",
            "Indian Code School"
        ]
        
        # Create nodes with influence metrics
        nodes = []
        channel_metrics = {}
        
        for channel in channels:
            # Generate realistic metrics for each channel
            influence = round(random.uniform(0.3, 0.9), 2)
            subscriber_factor = round(random.uniform(0.2, 0.8), 2)
            engagement_rate = round(random.uniform(0.4, 1.0), 2)
            
            channel_metrics[channel] = {
                'influence': influence,
                'subscriber_factor': subscriber_factor,
                'engagement_rate': engagement_rate
            }
            
            nodes.append({
                'data': {
                    'id': channel,
                    'label': channel,
                    'influence': influence,
                    'subscriber_factor': subscriber_factor,
                    'engagement_rate': engagement_rate,
                    'size': 20 + (influence * 30)  # Node size based on influence
                }
            })
        
        # Create weighted edges based on channel relationships
        edges = []
        for i, source in enumerate(channels):
            # Each channel connects to 4-8 other channels
            num_connections = random.randint(4, 8)
            possible_targets = channels.copy()
            possible_targets.remove(source)
            
            targets = random.sample(possible_targets, min(num_connections, len(possible_targets)))
            
            for target in targets:
                # Calculate edge weight based on both channels' metrics
                source_metrics = channel_metrics[source]
                target_metrics = channel_metrics[target]
                
                # Weight calculation formula:
                # - Higher weight for channels with similar influence levels
                # - Boost weight if both channels have high engagement
                influence_similarity = 1 - abs(source_metrics['influence'] - target_metrics['influence'])
                engagement_boost = (source_metrics['engagement_rate'] + target_metrics['engagement_rate']) / 2
                weight = round((influence_similarity * 0.7 + engagement_boost * 0.3), 2)
                
                edges.append({
                    'data': {
                        'source': source,
                        'target': target,
                        'weight': weight,
                        'width': 1 + (weight * 5),  # Edge width based on weight
                        'label': str(weight)  # Show weight as label
                    }
                })
        
        # Calculate top influencers based on weighted connections
        influence_scores = {}
        for edge in edges:
            source = edge['data']['source']
            weight = edge['data']['weight']
            influence_scores[source] = influence_scores.get(source, 0) + weight
        
        top_influencers = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)[:6]
        top_influencer_names = [name for name, _ in top_influencers]
        
        return jsonify({
            'nodes': nodes,
            'edges': edges,
            'influencers': top_influencer_names
        })
        
    except Exception as e:
        print(f"Error in youtube_network: {str(e)}")
        return jsonify({'error': str(e), 'nodes': [], 'edges': []})

@app.route('/traffic-spikes')
def traffic_spikes_page():
    return render_template('traffic_spikes.html')

@app.route('/api/simulate-traffic')
def simulate_traffic():
    traffic = traffic_manager.simulate_traffic_spike()
    is_spike = traffic_manager.detect_spike()
    
    # Distribute traffic using Round Robin
    distribution = traffic_manager.round_robin_balance(traffic)
    
    # Simulate Anycast routing
    client_regions = ["US", "EU", "ASIA"]
    anycast_routes = []
    for region in client_regions:
        server = traffic_manager.anycast_route(region)
        if server:
            anycast_routes.append({
                "region": region,
                "routed_to": server["id"],
                "server_location": server["location"]
            })
    
    return jsonify({
        "traffic_volume": traffic,
        "is_spike": is_spike,
        "round_robin_distribution": distribution,
        "anycast_routes": anycast_routes,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

@app.route('/api/server-status')
def get_server_status():
    return jsonify({
        "servers": traffic_manager.get_server_status(),
        "anycast_enabled": traffic_manager.anycast_enabled,
        "spike_threshold": traffic_manager.spike_threshold
    })

@app.route('/api/traffic-history')
def get_traffic_history():
    history = list(traffic_manager.traffic_history)
    formatted_history = [{
        "timestamp": h["timestamp"].strftime("%H:%M:%S"),
        "traffic": h["traffic"]
    } for h in history[-20:]]  # Last 20 entries
    
    return jsonify(formatted_history)

@app.route('/api/reset-loads')
def reset_loads():
    traffic_manager.reset_server_loads()
    return jsonify({"status": "success", "message": "Server loads reset"})

@app.route('/api/toggle-anycast')
def toggle_anycast():
    traffic_manager.anycast_enabled = not traffic_manager.anycast_enabled
    return jsonify({
        "anycast_enabled": traffic_manager.anycast_enabled,
        "status": "success"
    })


@app.route('/campaign-analysis')
def campaign_analysis_page():
    return render_template('campaign_analysis.html')

@app.route('/api/campaign-probability')
def campaign_probability():
    """Calculate campaign spread probability over time"""
    try:
        # Get campaign data from existing trend graphs
        campaigns = {}
        
        for trend, G in trend_graphs.items():
            if len(G.nodes()) > 5:  # Only consider trends with sufficient data
                # Calculate campaign metrics
                total_users = len(G.nodes())
                total_connections = len(G.edges())
                density = nx.density(G) if total_users > 1 else 0
                
                # Calculate spread probability using exponential growth model
                # P(t) = 1 - e^(-λt) where λ is spread rate
                spread_rate = min(0.8, density * total_users / 100)
                
                # Generate probability curve over 24 hours
                time_points = list(range(0, 25))  # 0 to 24 hours
                probabilities = [1 - math.exp(-spread_rate * t) for t in time_points]
                
                # Calculate campaign reach potential
                reach_potential = min(100, total_users * density * 10)
                
                campaigns[trend] = {
                    'total_users': total_users,
                    'total_connections': total_connections,
                    'density': round(density, 3),
                    'spread_rate': round(spread_rate, 3),
                    'time_points': time_points,
                    'probabilities': [round(p, 3) for p in probabilities],
                    'reach_potential': round(reach_potential, 1),
                    'final_probability': round(probabilities[-1], 3)
                }
        
        # Sort by reach potential
        sorted_campaigns = dict(sorted(campaigns.items(), 
                                     key=lambda x: x[1]['reach_potential'], 
                                     reverse=True)[:10])
        
        return jsonify({
            'campaigns': sorted_campaigns,
            'total_campaigns': len(campaigns),
            'avg_spread_rate': round(np.mean([c['spread_rate'] for c in campaigns.values()]), 3) if campaigns else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'campaigns': {}})

@app.route('/api/campaign-clusters')
def campaign_clusters():
    """Analyze campaign relationships using Boolean algebra and set theory"""
    try:
        # Create campaign clusters based on user overlap
        campaign_users = {}
        
        for trend, G in trend_graphs.items():
            if len(G.nodes()) > 3:
                campaign_users[trend] = set(G.nodes())
        
        if len(campaign_users) < 2:
            return jsonify({'clusters': {}, 'relationships': []})
        
        # Calculate set relationships
        relationships = []
        cluster_analysis = {}
        
        campaign_list = list(campaign_users.keys())
        
        for i, campaign1 in enumerate(campaign_list):
            for j, campaign2 in enumerate(campaign_list[i+1:], i+1):
                set1 = campaign_users[campaign1]
                set2 = campaign_users[campaign2]
                
                # Set operations
                intersection = set1 & set2
                union = set1 | set2
                difference1 = set1 - set2
                difference2 = set2 - set1
                symmetric_diff = set1 ^ set2
                
                # Calculate relationship metrics
                jaccard_similarity = len(intersection) / len(union) if union else 0
                overlap_coefficient = len(intersection) / min(len(set1), len(set2)) if set1 and set2 else 0
                
                # Boolean operations
                is_subset = set1.issubset(set2) or set2.issubset(set1)
                is_disjoint = set1.isdisjoint(set2)
                
                relationships.append({
                    'campaign1': campaign1,
                    'campaign2': campaign2,
                    'intersection_size': len(intersection),
                    'union_size': len(union),
                    'jaccard_similarity': round(jaccard_similarity, 3),
                    'overlap_coefficient': round(overlap_coefficient, 3),
                    'is_subset': is_subset,
                    'is_disjoint': is_disjoint,
                    'common_users': list(intersection)[:5]  # Show first 5 common users
                })
        
        # Create cluster analysis
        for campaign, users in campaign_users.items():
            cluster_analysis[campaign] = {
                'size': len(users),
                'users': list(users)[:10],  # Show first 10 users
                'density': round(len(users) / max(len(campaign_users.values()), key=len), 3)
            }
        
        # Sort relationships by similarity
        relationships.sort(key=lambda x: x['jaccard_similarity'], reverse=True)
        
        return jsonify({
            'clusters': cluster_analysis,
            'relationships': relationships[:15],  # Top 15 relationships
            'total_relationships': len(relationships)
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'clusters': {}, 'relationships': []})

@app.route('/api/probability-distribution')
def probability_distribution():
    """Generate probability distribution analysis"""
    try:
        # Analyze spread patterns using normal distribution
        all_densities = []
        all_sizes = []
        
        for trend, G in trend_graphs.items():
            if len(G.nodes()) > 2:
                density = nx.density(G)
                size = len(G.nodes())
                all_densities.append(density)
                all_sizes.append(size)
        
        if not all_densities:
            return jsonify({'error': 'No data available for analysis'})
        
        # Calculate distribution statistics
        density_mean = np.mean(all_densities)
        density_std = np.std(all_densities)
        size_mean = np.mean(all_sizes)
        size_std = np.std(all_sizes)
        
        # Generate normal distribution curves
        x_density = np.linspace(0, 1, 100)
        y_density = stats.norm.pdf(x_density, density_mean, density_std)
        
        x_size = np.linspace(0, max(all_sizes), 100)
        y_size = stats.norm.pdf(x_size, size_mean, size_std)
        
        return jsonify({
            'density_distribution': {
                'x': x_density.tolist(),
                'y': y_density.tolist(),
                'mean': round(density_mean, 3),
                'std': round(density_std, 3)
            },
            'size_distribution': {
                'x': x_size.tolist(),
                'y': y_size.tolist(),
                'mean': round(size_mean, 1),
                'std': round(size_std, 1)
            },
            'statistics': {
                'total_campaigns': len(all_densities),
                'avg_density': round(density_mean, 3),
                'avg_size': round(size_mean, 1),
                'density_variance': round(density_std**2, 4),
                'size_variance': round(size_std**2, 1)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/boolean-operations')
def boolean_operations():
    """Perform Boolean algebra operations on campaign sets"""
    campaign1 = request.args.get('campaign1')
    campaign2 = request.args.get('campaign2')
    
    if not campaign1 or not campaign2:
        return jsonify({'error': 'Please provide both campaign names'})
    
    try:
        # Get campaign user sets
        set1 = set(trend_graphs.get(campaign1, nx.Graph()).nodes()) if campaign1 in trend_graphs else set()
        set2 = set(trend_graphs.get(campaign2, nx.Graph()).nodes()) if campaign2 in trend_graphs else set()
        
        if not set1 or not set2:
            return jsonify({'error': 'One or both campaigns not found'})
        
        # Perform Boolean operations
        operations = {
            'intersection': list(set1 & set2),
            'union': list(set1 | set2),
            'difference_a_minus_b': list(set1 - set2),
            'difference_b_minus_a': list(set2 - set1),
            'symmetric_difference': list(set1 ^ set2),
            'is_subset_a_in_b': set1.issubset(set2),
            'is_subset_b_in_a': set2.issubset(set1),
            'is_disjoint': set1.isdisjoint(set2),
            'complement_visualization': {
                'set1_only': len(set1 - set2),
                'set2_only': len(set2 - set1),
                'intersection': len(set1 & set2),
                'universe_size': len(set1 | set2)
            }
        }
        
        return jsonify({
            'campaign1': campaign1,
            'campaign2': campaign2,
            'set1_size': len(set1),
            'set2_size': len(set2),
            'operations': operations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)