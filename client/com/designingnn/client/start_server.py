from flask import Flask, json
import socket
import argparse
import socket
import requests
from requests.exceptions import ConnectionError

app = Flask(__name__)


@app.route('/')
def test_endpoint():
    return "This is a test endpoint!!"


@app.route('/status')
def get_status():

    data = {
        'free': True
    }

    return app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )


def ping_server(host, port):
    try:
        r = requests.get(url="http://{}:{}".format(host, port))
        return r.status_code
    except ConnectionError:
        return 502


def register_client(server_host, server_port, client_port):
    data = {
        'host': socket.gethostname(),
        'port': int(client_port)
    }

    try:
        r = requests.post(url="http://{}:{}/register".format(server_host, server_port), data=data)
        return r.status_code
    except ConnectionError:
        return 502


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-server_port', help='The port on which Server can be contacted')
    parser.add_argument('-server_host', help='The hostname/IP on which Server is reachable')
    parser.add_argument('-client_port', help='The port on which the current client has to run')
    parser.add_argument('-data_dir', help='The directory on the client machine on which dataset exists')

    args = parser.parse_args()

    server_ping_status_code = ping_server(args.server_host, args.server_port)

    if server_ping_status_code == 200:
        print("server ping successful.")
        register_client(args.server_host, args.server_port, args.client_port)
        print("registered client to server.")

        print("starting client on port {}".format(args.client_port))
        app.run(host="0.0.0.0", port=int(args.client_port))
    else:
        print("Server ping failed. status code returned {}".format(server_ping_status_code))
