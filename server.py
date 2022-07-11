import json
from flask import Flask, request, jsonify

from main import request_simplification

app = Flask(__name__)

@app.route("/simplify", methods=['POST'])
def simplify():
    payload = request.get_json(silent=True)

    print(payload["sentence"])
    print(payload["path"])
    print(payload["bigram_factor"])
    print(payload["transformer"])

    old_tokens, new_tokens = request_simplification(payload["sentence"], payload["path"],
                                                    payload["bigram_factor"], payload["transformer"])

    result = []
    for i in range(len(old_tokens)):
        result.append({
            "old": old_tokens[i][0],
            "new": new_tokens[i][0]
        })

    return jsonify(result), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)