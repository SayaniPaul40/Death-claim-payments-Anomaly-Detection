from flask import Flask, request, jsonify
import json

from death_claim_usecase.main import Main as Death_claim

# Your existing code here...

app = Flask(__name__)

@app.route('/death_claim', methods=['POST'])
def death_claim():
    try:
        json_input = request.json

        d = Death_claim()

        death_claim= d.predict(json_input)

        return jsonify(death_claim)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
    
