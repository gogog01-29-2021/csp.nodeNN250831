# 1‑1‑1 Neural Network Online Trainer using CSP
#
# This module defines a simple neural network with a single scalar input,
# one hidden ReLU neuron, and a single scalar output.  The network is trained
# in an online (event‑wise) fashion using stochastic gradient descent within
# a Continuous Stream Processing (CSP) framework.
#
# The forward pass of the network is:
#
#     z1 = w1 * x + b1
#     h1 = max(0, z1)           # ReLU activation
#     ŷ  = w2 * h1 + b2
#     L   = 0.5 * (ŷ  - y)²    # mean squared error loss
#
# For each event containing an input value x and a true label y, the node
# computes the prediction ŷ , the loss, and then updates the weights (w1, b1, w2,
# b2) using gradient descent.  The gradients are derived analytically from
# the above expressions.
#
# See README.md for a full derivation of the gradients and example usage.

import csp
from csp import ts
from datetime import datetime, timedelta


@csp.node
def sgd_111(
    x: ts[float],
    y_true: ts[float],
    lr: float = 0.05,
) -> csp.Outputs(
    y_pred=ts[float],
    loss=ts[float],
    w1=ts[float],
    b1=ts[float],
    w2=ts[float],
    b2=ts[float],
):
    """
    A CSP node implementing a 1‑1‑1 neural network (scalar input, one hidden ReLU
    neuron, scalar output) with online stochastic gradient descent.

    Parameters
    ----------
    x : csp.ts[float]
        The input stream of scalar values.
    y_true : csp.ts[float]
        The stream of true target values corresponding to each input.
    lr : float, optional
        Learning rate for the SGD updates.  Defaults to 0.05.

    Outputs
    -------
    y_pred : csp.ts[float]
        The predicted output stream.
    loss : csp.ts[float]
        The half squared error loss for each event.
    w1, b1, w2, b2 : csp.ts[float]
        The current values of the model parameters after each update.

    This node maintains internal state for the weights and updates them whenever
    both the input and target streams tick and are valid.  It emits the
    prediction, loss, and updated weights as outputs.
    """

    # Initialize mutable state variables for weights and biases.
    with csp.state():
        s_w1 = 0.0
        s_b1 = 0.0
        s_w2 = 0.0
        s_b2 = 0.0

    # Set initial values for the weights at the start of the graph execution.
    with csp.start():
        # A small non‑zero weight helps avoid dead ReLUs during early training.
        s_w1 = 0.1
        s_b1 = 0.0
        s_w2 = 0.1
        s_b2 = 0.0

    # The node should update its state whenever both x and y_true tick and are valid.
    if csp.ticked(x, y_true) and csp.valid(x, y_true):
        # Forward pass
        z1 = s_w1 * x + s_b1
        h1 = z1 if z1 > 0.0 else 0.0  # ReLU activation
        yhat = s_w2 * h1 + s_b2
        err = yhat - y_true
        loss = 0.5 * err * err

        # Gradient computation (backpropagation)
        dy = err                        # ∂L/∂yhat
        grad_w2 = h1 * dy               # ∂L/∂w2
        grad_b2 = dy                    # ∂L/∂b2

        dh = s_w2 * dy                  # ∂L/∂h1
        dz = dh if z1 > 0.0 else 0.0    # ∂L/∂z1 (ReLU derivative)
        grad_w1 = x * dz                # ∂L/∂w1
        grad_b1 = dz                    # ∂L/∂b1

        # Update weights (SGD step)
        s_w2 -= lr * grad_w2
        s_b2 -= lr * grad_b2
        s_w1 -= lr * grad_w1
        s_b1 -= lr * grad_b1

        # Emit outputs
        return csp.output(
            y_pred=yhat,
            loss=loss,
            w1=s_w1,
            b1=s_b1,
            w2=s_w2,
            b2=s_b2,
        )


@csp.graph
def train_111_graph():
    """
    Construct a minimal training graph for the 1‑1‑1 network using synthetic data.

    This graph emits a few sample (x, y) pairs at one‑second intervals and feeds
    them into the `sgd_111` node.  The node updates its weights on each tick and
    prints the predictions, loss, and parameter values for inspection.
    """
    start = datetime(2020, 1, 1)

    # Synthetic input stream
    xs = csp.curve(float, [
        (start + timedelta(seconds=0), 0.50),
        (start + timedelta(seconds=1), -0.10),
        (start + timedelta(seconds=2), 0.80),
        (start + timedelta(seconds=3), 0.20),
    ])

    # Corresponding targets computed as 1.5 * ReLU(2x + 0.5)
    ys = csp.curve(float, [
        (start + timedelta(seconds=0), 1.5 * max(0.0, 2*0.50 + 0.5)),
        (start + timedelta(seconds=1), 1.5 * max(0.0, 2*(-0.10) + 0.5)),
        (start + timedelta(seconds=2), 1.5 * max(0.0, 2*0.80 + 0.5)),
        (start + timedelta(seconds=3), 1.5 * max(0.0, 2*0.20 + 0.5)),
    ])

    out = sgd_111(xs, ys, lr=0.05)

    # Print outputs for monitoring
    csp.print("y_pred", out.y_pred)
    csp.print("loss", out.loss)
    csp.print("w1", out.w1)
    csp.print("b1", out.b1)
    csp.print("w2", out.w2)
    csp.print("b2", out.b2)

# To run this example, you can execute:
#   csp.run(train_111_graph, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=4))
# from a Python interpreter after installing the `csp` library.