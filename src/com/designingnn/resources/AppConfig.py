import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# Agent config
##################################################################
##################################################################


DEFAULT_EPSILON = config['policy']['default_epsilon']

print("""
Starting execution with the following rl properties:

    Epsilon: {}

""".format(
    DEFAULT_EPSILON
))
