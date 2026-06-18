from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def receive_signal():
    data = request.json
    print("Received signal:", data)
    return jsonify({"status": "received"}), 200

@app.route('/')
def home():
    return "Webhook listener is running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)