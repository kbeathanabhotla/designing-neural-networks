
class Main:
    def __init__(self):
        pass

    def start_client(self, server_host, server_port, gpu_to_use):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-server_host', help='The hostname or IP of Server')
    parser.add_argument('-server_port', help='The port on which Server can be contacted')
    parser.add_argument('-gpu_to_use', help='The GPU number to use for this current client')
    parser.add_argument('-client_port', help='The port on which client can reached for progress or to assign run tasks')

    args = parser.parse_args()

    server_host = args.server_host
    server_port = args.server_port
    gpu_to_use = args.gpu_to_use
    client_port = args.client_port

    Main().start_client(server_host, server_port, gpu_to_use)
