# ochanomizu

# Session 4 :

## Perceptron AND Gate

This project implements an **AND gate** using a **simple perceptron** in Haskell with the HaskTorch library.
A simple perceptron is a type of neural network that maps data from an input layer to the output layer.

---

### Body

The initial code template was provided on the _Bekki Notion_ site.

It contained the following functions to implement:

- `step` — A Heaviside step function that returns `0` if input < 0, else `1`
- `perceptron` — A simple neural network that performs binary classification using weights and bias
- `calculateError` — Computes the error between the expected output and the predicted output

With the help of the HaskTorch documentation and **Valentin CHAUD**, we also completed the template :

- `train` — A training loop that updates weights and bias using the perceptron
- `main` — Sets up the data, trains the model, and displays final weights, bias, and performance

---

### Expectated results

The perceptron has to succesfully learn the behavior of an **AND gate** following this pattern :

- `AND(0,0) = 0`
- `AND(0,1) = 0`
- `AND(1,0) = 0`
- `AND(1,1) = 1`

---

### Training results

The initial error before training was 3.0.
In most runs the final output matched the expected output.
After running the training a few time the final error is 0.0 indicating that the perceptron has learned the pattern of the AND Gate.
A few runs showed an error of 1.0

After running the training multiple time, the perceptron was able to succesfully return the right result most of the time, however sometimes the predicted output didn't match the expected output. I cannot explain, nor can I find the source of this problem.

This issue was fixed by removing a stopping condition.
Now all in all runs the final output matches the expected output.
The final error is 0.0 the perceptron has fully learned the pattern of the AND Gate.

I wasn't able to fully produce the code of the train function and the main by myself due to a lack of experience in Haskell.
I required the help of my colleague **Valentin CHAUD** please refer to his README for more detail on those.

---

## Multi-layer perceptron XOR Gate

This projects implements a **XOR gate** using a **multiple-layer perceptron** in Haskell using the Torch library.
A multi-layer perceptron (MLP) is a type of neural network composed of multiple layers of interconnected neurons.
It consist on an input layer sending the data to one (or more) hidden layer until it get to the output layer.

---

### Body

The first code defines the MLP using a structure that defines the spcifications (how many neurons per layer & what function to use between layers) and one that defines the network (the list of layer & the function used between the layers).
Each training batch has 2 samples and trains for 2000 iterations.

It implements a MLP with 2 inputs, 1 hidden layer with 2 neuron and 2 ouput.
The code uses the function tanh, the MSE as the loss function and a gradiant descent function to update the weight.

The second code uses the function sigmuid instead of the tanh.

---

### Expectated results

The MLP has to succesfully learn the behavior of an **XOR gate** following this pattern :

- `XOR(0,0) = 0`
- `XOR(0,1) = 1`
- `XOR(1,0) = 1`
- `XOR(1,1) = 0`

---

## Author

Lyne PHAN A
