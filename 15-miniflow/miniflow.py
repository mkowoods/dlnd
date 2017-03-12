import numpy as np


class Node(object):

    def __init__(self, inbound_nodes = []):

        self.inbound_nodes = inbound_nodes

        #loop through each inbound and node and updates it outbound node array
        self.outbound_nodes = []
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)

        self.value = None
        self.gradients = {}


    def forward(self):
        """
        Forward Propragation

        compute the output values based on the inbound nodes and store the resutls
        in self.value

        """
        raise NotImplemented


    def backward(self):

        raise NotImplementedError

class Input(Node):

    def __init__(self):
        Node.__init__(self)

    def forward(self, value = None):
        if value is not None:
            self.value = value

    def backward(self):
        # An input node has no Inbound nodes so it's gradient (derivative) is zero
        # dervative of a constant is zero

        self.gradients = {self : 0}

        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.

        for node in self.outbound_nodes:
            grad_cost =  node.gradients[self]
            self.gradients[self] += grad_cost * 1.0

class Add(Node):

    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        tmp_sum = 0.0
        for node in self.inbound_nodes:
            tmp_sum += node.value
        self.value = tmp_sum


class Mul(Node):

    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        if len(self.inbound_nodes) == 0:
            return 0.0
        tmp_sum = 1.0
        for node in self.inbound_nodes:
            tmp_sum *= node.value
        self.value = tmp_sum


class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

        # NOTE: The weights and bias properties here are not
        # numbers, but rather references to other nodes.
        # The weight and bias values are stored within the
        # respective nodes.

    def forward(self):
        """
        Set self.value to the value of the linear function output.

        Your code goes here!
        """
        x, w, b = self.inbound_nodes
        self.value = np.dot(x.value, w.value) + b.value

    def backward(self):
        #IB Nodes are inputs, weights and bias

        inputs, weights, bias = self.inbound_nodes
        self.gradients = {node : np.zeros_like(node.value) for node in self.inbound_nodes}


        for node in self.outbound_nodes:
            grad_cost =  node.gradients[self]
            self.gradients[inputs]  += np.dot(grad_cost, weights.value.T)
            self.gradients[weights] += np.dot(inputs.value.T, grad_cost)
            self.gradients[bias]    += np.sum(grad_cost, axis=0, keepdims=False)



class Sigmoid(Node):

    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def forward(self):

        assert len(self.inbound_nodes) == 1
        self.value = -1
        self.value = self._sigmoid( self.inbound_nodes[0].value )


    def backward(self):
        ib_node = self.inbound_nodes[0]
        self.gradients = {ib_node: np.zeros_like(ib_node.value)}

        for node in self.outbound_nodes:
            grad_cost = node.gradients[self]
            deriv = self.value * (1 - self.value)
            self.gradients[ib_node] += (grad_cost * deriv)



class MSE(Node):

    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        #m = 1.0
        self.m = y.shape[0]
        self.diff = (y - a)
        self.value = np.sum(np.square(self.diff))/self.m

    def backward(self):
        y, a = self.inbound_nodes
        self.gradients[y] = ( 2./self.m) * self.diff #partial derviative is responsible for the sign flipping
        self.gradients[a] = (-2./self.m) * self.diff


def topological_sort(feed_dict):

    input_nodes = feed_dict.keys()

    #run DFS to build graph G
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        node = nodes.pop(0)

        if node not in G:
            G[node] = {'in': set(), 'out': set()}
        for ob_node in node.outbound_nodes:
            if ob_node not in G:
                G[ob_node] = {'in': set(), 'out': set()}
            G[node]['out'].add(ob_node)
            G[ob_node]['in'].add(node)
            nodes.append(ob_node)

    #run Kahn's algorithm to get topo sort

    L = []
    S = set([n for n in input_nodes])
    while len(S) > 0:
        node = S.pop()
        if isinstance(node, Input):
            node.value = feed_dict[node]
        L.append(node)

        #remove edge from node to outbound nodes
        for m in node.outbound_nodes:
            G[node]['out'].remove(m)
            G[m]['in'].remove(node)

            #if there are no more inbound nodes then add it to the frontier
            if len(G[m]['in']) == 0:
                S.add(m)
    return L



def forward_pass(output_node = None, sorted_nodes = []):
    """
    takes a single output node and a list of sorted nodes from topological sort

    :param output_node:
    :param sorted_nodes:
    :return:
    """
    for node in sorted_nodes:
        node.forward()
    return output_node.value


def forward_and_backward(sorted_notes):

    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for node in sorted_notes:
        node.forward()
        #print(node, node.value)

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for node in sorted_notes[::-1]:
        node.backward()


def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` Nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # TODO: update all the `trainables` with SGD
    # You can access and assign the value of a trainable with `value` attribute.
    # Example:
    # for t in trainables:
    #   t.value = your implementation here

    for t in trainables:
        t.value += -learning_rate * t.gradients[t]

if __name__ == "__main__":

    x,y = Input(), Input()

    add = Add(x, y)

    feed_dict = {x : 10, y : 20}