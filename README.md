# Hyper-parameter

Hyper-parameters to tune:

- $\alpha$
- $\beta$
- $\beta_1, \beta_2, \epsilon$
- layers
- hidden units
- learning rate decay


# Coarse-to-fine

Start with random numbers and identify region of parameters that seem to have the most impact then zoom in on that region.

# Learning rate

Sample at uniformly from random on a log scale from .0001 and 1.

```python
r = -4 * np.random.rand() -> [-4,0]
alpha = 10**r -> [10^-4,10^0]
```
# Exponentially weighted averages

Similarly, sample $\beta$ from a logarithmic range.

Let's say we are trying to find the optimal $\beta$ between .9 and .999

We will sample $1-\beta$ from among .1 to .001 on the log scale.

```python
r = -2 * np.random.rand()-1 -> [-3,-1]
beta = 1 - 10**r -> [10^-4,10^0]
```

# Batch normalization

## Motivation

Normalizing data helps learning proceed faster. The motivation behind batch normalization is to normal the intermediate values in each layer in addition to the initial data.


## Formula

Let the normalized version of each intermediate value $z^{(i)}$ be:

$z_{norm}^{(i)} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$

To generalize this to any distribution, we can rescale using two parameters:

$\tilde{z}^{(i)} = \gamma z_{norm}^{(i)} + \beta$, where $\gamma$ and $\beta$ are learnable parameters.

So instead of $z^{(i)}$, we would use $z^{(i)}_{norm}$ to calculate the activation values $a^{(i)}$

** Note that if $\gamma = \sqrt{\sigma^2 + \epsilon}$ and $\beta = \mu$ we have $\tilde{z}^{(i)} = z^{(i)}$

In practice, batch normalization is applied on mini-batches and built into deep learning frameworks.

Note that since batch normalization centers the values and zeros out the means, the constant terms $b$ actually get cancelled out. So our forward propagation actually gets simplified to:

$z^{[l]} = w^{[l]}a^{[l-1]}$

The parameters $\gamma$ and $\beta$ are adjusted during back propagation in a similar manner to the weights $W$

## Effect

In addition to making learning faster, batch normalization also helps neural networks become robust to changes in input data. Because values in earlier layers are broadcast to subsequent layers, changes in input data can cause significant changes in the activation values that later layers to have to learn on. By doing batch normalization, the intermediate values are more stable and reduce the effect of changes in input data. It also helps reduce coupling between layers and help layers become more independent.

## Test time

In test time, we sometimes might need to process only a single sample and there is no meaning to taking the mean and variance of a single example.

What we do is then take the mean and variance from the training phase either by:

- Running the whole training set through the final training model to calculate mean and variance
- Calculate exponentially weighted mean and variance through each layer.

# Softmax Clasifier


## Activation function
We use the softmax activation function for multi-class classification.

Let:
$$ t = e^{z^{[l]}} $$

The activation values for n different classes are:
$$ a^{[l]} = \frac{t}{\sum_{j=1}^n t_i}$$

The softmax functions gives us the probabilities of the training example being each of the n classes.

For example:

$$z^{[l]} = [5, 2, -1, 3]$$

then

$$t = [e^5, e^2, e^{-1}, e^3] = [148.4, 7.4, 0.4, 20.1]$$

and

$$\sum_{j=1}^4 t_i = 176.3$$

The activation values are then $a^{[l]} = \frac{t}{176.3} = [.842, .042, .002, .114]$

## Loss function

Let:

$$ y^{(i)} = [0, 1, 0, 0] $$

and the corresponding activation values be:

$$ a^{[l](i)} = [.3, .2, .1, 4] $$

In this particular example, our algorithm is not doing very well since it only predicts 20% chance of the true label.

The loss function for a single training example in multi-class classifier is:

$$l(\hat{y}, y) = -\sum_{j=1}^4 y_j \log \hat{y}_j$$

This essentially reduces to the $\log \hat{y}_j$ value of the true label - $\log 0.2$ - since $y_1, y_3, y_4$ are all 0 and $y_2$ is 1.

In order for $l$ to be small, $\hat{y}$ of the true label must be big, which is what we want in a classifier.

The total cost function is then:

$$L = - \frac{1}{m} \sum_{i=1}^m \sum_{j=1}^n y_j \log \hat{y}_j$$

## Gradient descent

The derivative of a softmax activation function is:

$$dz^{[L]} = \hat{y} - y$$
