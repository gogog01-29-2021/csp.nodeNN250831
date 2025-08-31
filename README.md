# csp.nodeNN250831
1‑1‑1 Neural Network Online Trainer

This repository contains an example of training a very simple neural network—one
with a single scalar input, a single hidden neuron with a ReLU activation, and
a single scalar output—using online stochastic gradient descent. The code
is implemented using Point72's Continuous Stream Processing (CSP)
library
, which allows neural network updates
to be triggered by incoming events rather than by processing entire data
batches.

Network definition

The forward pass of a 1‑1‑1 network can be written as:

𝑧
1
	
=
𝑤
1
𝑥
+
𝑏
1
,


ℎ
1
	
=
R
e
L
U
(
𝑧
1
)
=
max
⁡
(
0
,
𝑧
1
)
,


𝑦
^
	
=
𝑤
2
ℎ
1
+
𝑏
2
,


𝐿
	
=
1
2
(
𝑦
^
−
𝑦
)
2
.
z
1
	​

h
1
	​

y
^
	​

L
	​

=w
1
	​

x+b
1
	​

,
=ReLU(z
1
	​

)=max(0,z
1
	​

),
=w
2
	​

h
1
	​

+b
2
	​

,
=
2
1
	​

(
y
^
	​

−y)
2
.
	​


Here, 
𝑤
1
,
𝑏
1
w
1
	​

,b
1
	​

 are the input–hidden layer parameters, 
𝑤
2
,
𝑏
2
w
2
	​

,b
2
	​

 are the
hidden–output parameters, 
𝑥
x is the input, 
𝑦
y is the true target, and
ReLU is the rectified linear unit activation.

Gradients for online learning

When a single training example 
(
𝑥
,
𝑦
)
(x,y) arrives, we compute the prediction

𝑦
^
y
^
	​

 and then update the weights using stochastic gradient descent.
The relevant partial derivatives are:

𝛿
𝑦
	
:
=
∂
𝐿
∂
𝑦
^
=
𝑦
^
−
𝑦
,


∂
𝐿
∂
𝑤
2
	
=
ℎ
1
  
𝛿
𝑦
,
	
∂
𝐿
∂
𝑏
2
	
=
𝛿
𝑦
,


𝛿
ℎ
	
:
=
∂
𝐿
∂
ℎ
1
=
𝑤
2
  
𝛿
𝑦
,


𝛿
𝑧
	
:
=
∂
𝐿
∂
𝑧
1
=
𝛿
ℎ
⋅
1
{
𝑧
1
>
0
}
,


∂
𝐿
∂
𝑤
1
	
=
𝑥
  
𝛿
𝑧
,
	
∂
𝐿
∂
𝑏
1
	
=
𝛿
𝑧
.
δ
y
	​

∂w
2
	​

∂L
	​

δ
h
	​

δ
z
	​

∂w
1
	​

∂L
	​

	​

:=
∂
y
^
	​

∂L
	​

=
y
^
	​

−y,
=h
1
	​

δ
y
	​

,
:=
∂h
1
	​

∂L
	​

=w
2
	​

δ
y
	​

,
:=
∂z
1
	​

∂L
	​

=δ
h
	​

⋅1{z
1
	​

>0},
=xδ
z
	​

,
	​

∂b
2
	​

∂L
	​

∂b
1
	​

∂L
	​

	​

=δ
y
	​

,
=δ
z
	​

.
	​


An online SGD update with learning rate 
𝜂
η then adjusts each weight
according to

𝑤
𝑖
←
𝑤
𝑖
−
𝜂
  
∂
𝐿
∂
𝑤
𝑖
,
𝑏
𝑖
←
𝑏
𝑖
−
𝜂
  
∂
𝐿
∂
𝑏
𝑖
,
w
i
	​

←w
i
	​

−η
∂w
i
	​

∂L
	​

,b
i
	​

←b
i
	​

−η
∂b
i
	​

∂L
	​

,

for 
𝑖
∈
{
1
,
2
}
i∈{1,2}.

Code structure

The implementation lives in sgd_111_csp.py and defines two key
components:

sgd_111 – a CSP node that encapsulates the network state
(weights and biases), performs the forward pass, computes gradients for a
single training example, and updates its state. It emits the prediction,
loss, and current parameter values on every update.

train_111_graph – a sample CSP graph that constructs synthetic
streams of inputs and targets, feeds them into sgd_111, and prints the
outputs. You can modify the curves to use your own data streams.

To run the example training graph for four seconds:

from datetime import datetime, timedelta
from sgd_111_csp import train_111_graph
import csp

csp.run(
    train_111_graph,
    starttime=datetime(2020, 1, 1),
    endtime=timedelta(seconds=4)
)


As the graph runs, it will print out the predicted value, the loss, and the
updated parameters after each event arrives. You should see the loss decrease
and the weights adjust toward values that fit the synthetic data.

Notes

CSP installation: The code depends on Point72's
csp
 library. You can install it with
pip install csp if it's not already available in your environment.

Event‑driven training: Unlike conventional mini‑batch training in deep
learning frameworks, this example updates the network parameters immediately
after each data point. This can be useful in streaming contexts where data
arrive continuously and the model must adapt on the fly.

Pushing to GitHub

This repository is intended as an example. If you wish to commit these files
to a GitHub repository, clone the repository locally and copy sgd_111_csp.py
and README.md into it. Then run:

git add sgd_111_csp.py README.md
git commit -m "Add 1‑1‑1 network example with online SGD"
git push


In this environment we are unable to push directly via the CLI due to network
restrictions, so please perform the final push from your own machine.
