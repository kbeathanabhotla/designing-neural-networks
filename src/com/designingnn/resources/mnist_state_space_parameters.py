output_states = 10
image_size = 28

layer_limit = 6

# Transition Options
possible_conv_depths = [64, 128, 256, 512]
possible_conv_sizes = [1, 3, 5]
possible_pool_sizes = [[5, 3], [3, 2], [2, 2]]
max_fc = 2
possible_fc_sizes = [i for i in [512, 256, 128] if i >= output_states]

allow_initial_pooling = False
init_utility = 0.5
allow_consecutive_pooling = False

conv_padding = 'SAME'

batch_norm = False

# Epislon schedule for q learning agent.
# Format : [[epsilon, # unique models]]
# Epsilon = 1.0 corresponds to fully random, 0.0 to fully greedy
epsilon_schedule = [[1.0, 1500],
                    [0.9, 100],
                    [0.8, 100],
                    [0.7, 100],
                    [0.6, 150],
                    [0.5, 150],
                    [0.4, 150],
                    [0.3, 150],
                    [0.2, 150],
                    [0.1, 150]]

# Q-Learning Hyper parameters
learning_rate = 0.01
discount_factor = 1.0
replay_number = 128


# Set up the representation size buckets (see paper Appendix Section B)
def image_size_bucket(image_size):
    if image_size > 7:
        return 8
    elif image_size > 3:
        return 4
    else:
        return 1


# Condition to allow a transition to fully connected layer based on the current representation size
def allow_fully_connected(representation_size):
    return representation_size <= 4
