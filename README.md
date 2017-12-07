# Baselines DNC branch

This branch adds the [Differentiable Neural Computer (DNC)](https://github.com/deepmind/dnc) to OpenAI Baselines algorithms, as well as some saving and plotting logic.


#### Seems to work with:

- a2c

#### Seems to not work with:

- ACKTR - The authors of the K-FAC optimizer mention that the model should only use 
[fully-connected or convolutional layers](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/kfac),
adding a second optimizer for the recurrent parts of the model seemed to not work.


#### Requires
- The DNC code must be linked in the project settings or PYTHONPATH,

- matplotlib and pandas for plotting