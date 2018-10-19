# Designing Neural Networks using Reinforcement learning

The goal of this application is to generate a CNN best suited for a problem using Reinforcement learning.


## Introduction
* Generally training and testing CNN architectures takes a long time even on GPUs.
* Hence we implement a *client-server* strategy. So that, multiple clients can connect to a server and the training and testing phases would be done on client and server just co-ordinates the entire cycle.
	* **Server**: Responsible to perform the following:
		* Generating new CNN architectures.
		* Sending the client an architecture and waiting for result.
		* Store the results to replay database and play them back to generate new architecture.
		* Show training progress on demand.

    * **Client**: Responsible to perform the following:
    	* Get a new network architecture.
    	* Train the network and report the accuracies back to server.
    	* Report progress to server on demand.

## How to run Server


## How to run Client

