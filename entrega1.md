# Replication project: "A Neural Representation of Sketch Drawings"

Jou-Hui Ho, Hojin Kang


### Description of the problem

The main problem to solve consists on the generation of hand-drawn sketch drawings of different classes of objects, from training examples of the game <i>Quick, Draw!</i>. The solution will be implemented using an encoder-decoder autoregressive model.

- <b>Input:</b> The input of the model is a dataset of hand-drawn sketches, each represented as a sequence of motor actions controlling a pen: the direction of the movement, when to lift the pen up, and when to stop drawing. More concretely, each input is a vector containing 5 elements: 
(\Delta x, \Delta y, p_1, p_2, p_3)

<img src="https://latex.codecogs.com/svg.latex?\Large&space;(\Delta x, \Delta y, p_1, p_2, p_3)" title="\Large (\Delta x, \Delta y, p_1, p_2, p_3)" />


where the first two elements are the offset distance from the previous point, and the last 3 elements represents a one-hot vector of the 3 mentioned states.
