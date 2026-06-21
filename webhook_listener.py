from flask import Flask, request, jsonify
import os
import json
from datetime import datetime

app = Flask(__name__)

LOG_FILE = "signals_log.jsonl"

@app.route('/webhook', methods=['POST'])
def receive_signal():
    data = request.get_json(force=True, silent=True)
    if data is None:
        raw = request.data.decode('utf-8')
        data = {"raw": raw}

    record = {
        "received_at": datetime.utcnow().isoformat(),
        "payload": data
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    print("Received signal:", record)
    return jsonify({"status": "received"}), 200

@app.route('/')
def home():
    return "Webhook listener is running."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)