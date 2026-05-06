import os
import json
import hashlib
from datetime import datetime

from flask import Flask, jsonify, request
from flask_cors import CORS
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# API Keys from environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "screentime-coach"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=128,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# Connect to OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)


def save_session(app_usage, ai_content):
    vector = []
    app_minutes = {app["name"]: app["minutes"] for app in app_usage["apps"]}
    apps_to_track = ["Instagram", "YouTube", "WhatsApp", "Chrome", "Twitter", "Swiggy"]
    for app_name in apps_to_track:
        vector.append(float(app_minutes.get(app_name, 0)))
    while len(vector) < 128:
        vector.append(0.0)

    session_id = f"session-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    apps_json = json.dumps(app_usage["apps"])
    suggestion_value = ai_content["suggestion"]
    if isinstance(suggestion_value, list):
        suggestion_value = json.dumps(suggestion_value)

    index.upsert(vectors=[{
        "id": session_id,
        "values": vector,
        "metadata": {
            "date": app_usage["date"],
            "total_minutes": app_usage["total_screen_time_minutes"],
            "pickups": app_usage.get("pickups", 0),
            "late_night_usage": app_usage.get("late_night_usage", False),
            "apps_json": apps_json,
            "nudge": ai_content["nudge"],
            "micro_story": ai_content["micro_story"],
            "suggestion": suggestion_value
        }
    }])
    return session_id


def get_past_sessions(app_usage):
    vector = []
    app_minutes = {app["name"]: app["minutes"] for app in app_usage["apps"]}
    apps_to_track = ["Instagram", "YouTube", "WhatsApp", "Chrome", "Twitter", "Swiggy"]
    for app_name in apps_to_track:
        vector.append(float(app_minutes.get(app_name, 0)))
    while len(vector) < 128:
        vector.append(0.0)

    results = index.query(vector=vector, top_k=3, include_metadata=True)
    past_sessions = []
    for match in results.matches:
        apps_list = []
        apps_json = match.metadata.get("apps_json")
        if apps_json:
            try:
                apps_list = json.loads(apps_json)
            except:
                apps_list = []
        past_sessions.append({
            "date": match.metadata.get("date"),
            "total_minutes": match.metadata.get("total_minutes"),
            "total_screen_time_minutes": match.metadata.get("total_minutes"),
            "pickups": match.metadata.get("pickups", 0),
            "late_night_usage": match.metadata.get("late_night_usage", False),
            "apps": apps_list,
            "nudge": match.metadata.get("nudge")
        })
    return past_sessions


def generate_content(app_usage, prev_avg_total=0, prev_avg_apps={}):
    app_list = ""
    for app in app_usage["apps"]:
        app_name = app['name']
        today_mins = app['minutes']
        prev_avg = prev_avg_apps.get(app_name, 0)
        if prev_avg > 0:
            change = today_mins - prev_avg
            pct = round(abs(change) / prev_avg * 100)
            direction = "up" if change > 0 else "down"
            app_list += f"- {app_name} ({app['category']}): {today_mins} mins (avg: {prev_avg} mins, {direction} {pct}%)\n"
        else:
            app_list += f"- {app_name} ({app['category']}): {today_mins} minutes\n"

    past_sessions = get_past_sessions(app_usage)
    past_context = ""
    if past_sessions:
        totals = [s['total_minutes'] for s in past_sessions if s['total_minutes']]
        avg_mins = int(sum(totals) / len(totals)) if totals else 0
        past_context = f"User's past {len(past_sessions)} sessions:\n"
        for session in past_sessions:
            past_context += f"- Date: {session['date']}, Total minutes: {session['total_minutes']}\n"
        past_context += f"Average screen time: {avg_mins} minutes\n"
        today_total = app_usage['total_screen_time_minutes']
        if avg_mins > 0:
            total_change_pct = round(abs(today_total - avg_mins) / avg_mins * 100)
            direction = "HIGHER" if today_total > avg_mins else "LOWER"
            past_context += f"Today's usage of {today_total} mins is {direction} than average by {total_change_pct}%\n"
        late_night_count = sum(1 for s in past_sessions if s.get('late_night_usage'))
        if late_night_count >= 2:
            past_context += f"Pattern: Late night usage in {late_night_count} of last {len(past_sessions)} sessions\n"

    prompt = f"""
    You are a friendly digital wellness coach who analyzes behavior patterns.
    The user's screen time today is {app_usage['total_screen_time_minutes']} minutes.
    Late night usage tonight: {app_usage['late_night_usage']}
    Phone pickups today: {app_usage['pickups']} times
    Apps used today (with comparison to past averages):
    {app_list}
    {past_context}
    Analyze the user's behavior deeply and respond ONLY with a valid JSON object:
    {{
        "nudge": "2-3 sentences comparing today's total screen time to their average with exact % change. Mention which specific apps increased or decreased and by how much %.",
        "micro_story": "4-5 sentence motivational story about someone who reduced their screen time and improved their life",
        "suggestion": ["tip 1 targeting their most overused app today", "tip 2 about their usage pattern (late night / pickups etc)", "tip 3 a general digital wellness habit"],
        "behavioral_insight": "1-2 sentences of deep behavioral observation"
    }}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    if isinstance(result.get("suggestion"), str):
        result["suggestion"] = [result["suggestion"]]
    return result


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "AI Screen-Time Coach API is running!"})


@app.route("/coach", methods=["GET"])
def get_coaching():
    return jsonify({"error": "Use POST /coach-android with usage data"}), 400


@app.route("/coach-android", methods=["POST"])
def get_coaching_android():
    app_usage = request.get_json()
    past_sessions = get_past_sessions(app_usage)
    prev_avg_total = 0
    prev_avg_apps = {}

    if past_sessions and len(past_sessions) > 0:
        totals = [s.get("total_screen_time_minutes", 0) for s in past_sessions if s.get("total_screen_time_minutes")]
        if totals:
            prev_avg_total = round(sum(totals) / len(totals))
        app_names = [a["name"] for a in app_usage["apps"]]
        for app_name in app_names:
            app_minutes_list = []
            for session in past_sessions:
                past_apps = session.get("apps", [])
                for pa in past_apps:
                    if pa.get("name") == app_name:
                        app_minutes_list.append(pa.get("minutes", 0))
            prev_avg_apps[app_name] = round(sum(app_minutes_list) / len(app_minutes_list)) if app_minutes_list else 0

    content = generate_content(app_usage, prev_avg_total, prev_avg_apps)
    save_session(app_usage, content)

    return jsonify({
        "usage": app_usage,
        "nudge": content["nudge"],
        "micro_story": content["micro_story"],
        "suggestion": content["suggestion"],
        "behavioral_insight": content.get("behavioral_insight", ""),
        "past_sessions_count": len(past_sessions),
        "prev_avg_total": prev_avg_total,
        "prev_avg_apps": prev_avg_apps
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
