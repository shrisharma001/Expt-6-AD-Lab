import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from textblob import TextBlob
from typing import List
import googleapiclient.discovery
import googleapiclient.errors

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
print("RUNNING EXP 6 FILE ✅")
# 🔑 PUT YOUR API KEY HERE
YOUTUBE_API_KEY = "AIzaSyDklZzehEQAnCvbkSkLAQc_iVZJ--QYZSM"

# Initialize YouTube client
youtube = None
try:
    youtube = googleapiclient.discovery.build(
        "youtube",
        "v3",
        developerKey=YOUTUBE_API_KEY
    )
    print("✓ YouTube API initialized successfully")
except Exception as e:
    print("❌ Error initializing YouTube API:", e)
    youtube = None


# ─────────────────────────────────────────────
# FastAPI Setup
# ─────────────────────────────────────────────

app = FastAPI(title="YouTube Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    url: str
    limit: int = 100


class CommentResponse(BaseModel):
    comment_id: str
    text: str
    sentiment: str
    polarity: float


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def extract_video_id(url: str):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_sentiment(text: str):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0.05:
        category = "Positive"
    elif polarity < -0.05:
        category = "Negative"
    else:
        category = "Neutral"

    return category, round(polarity, 4)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/health")
def health_check():
    return {
        "status": "Running",
        "youtube_api_configured": youtube is not None
    }


@app.post("/analyze_comments/", response_model=List[CommentResponse])
def analyze_comments(request: AnalysisRequest):

    if youtube is None:
        raise HTTPException(
            status_code=500,
            detail="YouTube API not initialized"
        )

    video_id = extract_video_id(request.url)

    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL"
        )

    comments = []
    next_page_token = None
    total_to_fetch = min(request.limit, 500)

    try:
        while len(comments) < total_to_fetch:

            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, total_to_fetch - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance"
            ).execute()

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                text = snippet["textDisplay"]
                cid = item["id"]

                sentiment, polarity = get_sentiment(text)

                comments.append(CommentResponse(
                    comment_id=cid,
                    text=text[:500],
                    sentiment=sentiment,
                    polarity=polarity
                ))

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        if not comments:
            raise HTTPException(
                status_code=404,
                detail="No comments found"
            )

        return comments

    except googleapiclient.errors.HttpError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ─────────────────────────────────────────────
# Run Server
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)