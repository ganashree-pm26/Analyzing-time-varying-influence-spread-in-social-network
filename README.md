# Analyzing-time-varying-influence-spread-in-social-network
A Flask-based web application that monitors Reddit and YouTube data to track how influence spreads over time. It identifies top influencers using CELF and TAIC algorithms, visualizes dynamic influence graphs, detects fake users based on real behavioral cues, and manages server load using Anycast + Round Robin traffic balancing.
## 🚀 Features

| Module                            | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| 🔥 Trending Topics               | Shows what's trending on Reddit over the last 48 hours using NLP.           |
| 📈 Time-Varying Influence Graphs | Displays influence spread for top trends using timestamped interaction data.|
| 🎯 Top Influencers (CELF)        | Identifies most influential users using the CELF greedy algorithm.          |
| ⏱️ TAIC Influencers             | Detects top influencers using time-aware influence decay model.             |
| 🧭 Influence Path (Dijkstra)     | Traces the shortest influence path between any two users.                   |
| 🧠 Fake News Detection           | Calculates a fake score for suspicious users based on spikes, HTTPS, etc.   |
| 🎬 YouTube Influence Graph       | Builds dynamic influence network of trending YouTube creators.              |
| 🌐 Load Balancer Simulation      | Simulates server traffic distribution using Anycast + Round Robin.          |

---

## 🛠️ Tech Stack

- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap, Cytoscape.js
- **Backend:** Python, Flask, NetworkX, NLTK
- **APIs Used:**
  - Reddit API (via `praw`)
  - YouTube Data API v3
- **Data Handling:**
  - Influence routing via Dijkstra's Algorithm
  - Influence detection via CELF & TAIC models
  - NLP with NLTK for Reddit topic parsing
- **Extras:**
  - Fake user detection logic (influence spikes, HTTPS check, duplicates)
  - Traffic flow management using simulated Round Robin and Anycast

---
## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ganashree-pm26/Analyzing-time-varying-influence-spread-in-social-network.git
cd Analyzing-time-varying-influence-spread-in-social-network
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # For Linux/macOS  
.\venv\Scripts\activate         # For Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Keys

Edit app.py and update:

YouTube API key → YOUTUBE_API_KEY = 'your_key_here'

Reddit client ID/secret → in praw.Reddit(...)


### 5. Run the application

```bash
python app.py
```
Then open http://127.0.0.1:5000 in your browser.

## 📈 Algorithms & Techniques Used

### ✅ CELF (Cost-Effective Lazy Forward)
Used to find **top-k influential users** with the highest expected influence spread via greedy optimization.

---

### ⏳ TAIC (Time-Aware Influence Calculation)
Models **real-world decay of influence over time**.  
Applies **exponential decay** based on timestamp difference between interactions.

---

### 📍 Dijkstra’s Algorithm
Finds the **shortest influence route** between any two users in a directed weighted graph.

---

### 🧠 NLP + POS Tagging
Uses **NLTK** to extract **keywords and nouns** from Reddit titles.  
Filters out non-influential words using **stopword removal** and **POS tagging**.

---

### ⚠️ Fake Score Detection
Each user is dynamically scored out of 10 based on:

- ⚡ Sudden influence spike in **< 5 minutes**
- 🔓 Source URL **does not use HTTPS**
- 📄 Content **duplication from another user**

---

## 💡 Why this Project?

In the age of **rapid information sharing**, understanding how influence spreads and detecting fake news early is vital.

This tool helps visualize:

- 👥 Who influences who  
- 🔥 Which topics trend  
- 🚨 Which accounts spread misinformation

It combines **graph theory**, **real-time data**, and **AI-powered NLP** in one platform.

---

## 🔭 Future Work

- 🔐 Add login/auth system for **user-specific dashboards**
- ⚙️ Add **GraphQL backend APIs**
- 📱 Expand to platforms like **Twitter** or **Instagram**
- 🧑‍🤝‍🧑 Enable **community-based flagging** of fake users

---

## 👩‍💻 Author

**Ganashree PM** – Built for **academic** + **real-world research** on social media influence analysis.
