import argparse


class Main:
    def __init__(self):
        pass

    def start_server(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-server_port', help='The port on which Server can be contacted')
    parser.add_argument('-repo_dir', help='The directory on ther server which can be used as a repo')

    args = parser.parse_args()

    print(args)

    Main().start_server()
