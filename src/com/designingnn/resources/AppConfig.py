import configparser

config = configparser.ConfigParser()
config.read('config.ini')

DEFAULT_EPSILON = config['policy']['default_epsilon']


print("""
Starting execution with the following properties:

    Epsilon: {}

""".format(
    DEFAULT_EPSILON
))
