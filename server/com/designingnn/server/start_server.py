from flask import Flask, request
import sys
import json

app = Flask(__name__)


@app.route('/')
def test_endpoint():
    return "This is a test endpoint!!"


@app.route('/register', methods=['POST'])
def register_client():
    print("registering client with payload {}".format(request.data))
    data = json.loads(request.data)

    print data

    data['status'] = 'registered'
    # return redirect(url_for('success', name=user))

    return app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(sys.argv[1]))
