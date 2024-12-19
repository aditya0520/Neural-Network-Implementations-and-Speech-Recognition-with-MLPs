import numpy as np
from mytorch.functional_hw1 import *


class Activation(object):
    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result,
    i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an
    # abstract base class for the others

    # Note that these activation functions are scalar operations. I.e, they
    # shouldn't change the shape of the input.

    def __init__(self, autograd_engine):
        self.state = None
        self.autograd_engine = autograd_engine

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError


class Identity(Activation):
    """
    Identity function (already implemented).
    This class is a gimme as it is already implemented for you as an example.
    Just complete the forward by returning self.state.
    """
    def __init__(self, autograd_engine):
        super(Identity, self).__init__(autograd_engine)

    def forward(self, x):

        self.state = x

        return self.state


class Sigmoid(Activation):
    """
    Sigmoid activation.
    Feel free to enumerate primitive operations here (you may find it helpful).
    Feel free to refer back to your implementation from part 1 of the HW for equations.
    """
    def __init__(self, autograd_engine):
        super(Sigmoid, self).__init__(autograd_engine)

    def forward(self, x):

        minus_one = np.ones_like(x) * -1
        one = np.ones_like(x)

        neg_x = x * -1
        self.autograd_engine.add_operation(
            inputs=[x, minus_one],
            output=neg_x,
            gradients_to_update=[None, None], 
            backward_operation=mul_backward
        )

        exp_neg_x = np.exp(neg_x)
        self.autograd_engine.add_operation(
            inputs=[neg_x],
            output=exp_neg_x,
            gradients_to_update=[None],  
            backward_operation=exp_backward
        )

        one_plus_exp_neg_x = 1 + exp_neg_x
        self.autograd_engine.add_operation(
            inputs=[one, exp_neg_x],
            output=one_plus_exp_neg_x,
            gradients_to_update=[None, None],  
            backward_operation=add_backward
        )

        sigmoid_output = 1 / one_plus_exp_neg_x
        self.autograd_engine.add_operation(
            inputs=[one, one_plus_exp_neg_x],
            output=sigmoid_output,
            gradients_to_update=[None, None],  
            backward_operation=div_backward
        )
        self.state = sigmoid_output
        
        return sigmoid_output



class Tanh(Activation):
    """
    Tanh activation.
    Feel free to enumerate primitive operations here (you may find it helpful).
    Feel free to refer back to your implementation from part 1 of the HW for equations.
    """
    def __init__(self, autograd_engine):
        super(Tanh, self).__init__(autograd_engine)

    def forward(self, x):

        # TODO Compute forward with primitive operations
        # TODO Add operations to the autograd engine as you go

        minus_one = np.ones_like(x) * -1
        one = np.ones_like(x)
        
        neg_x = x * minus_one
        self.autograd_engine.add_operation(
            inputs=[x, minus_one],
            output=neg_x,
            gradients_to_update=[None, None], 
            backward_operation=mul_backward
        )

        exp_neg_x = np.exp(neg_x)
        self.autograd_engine.add_operation(
            inputs=[neg_x],
            output=exp_neg_x,
            gradients_to_update=[None],  
            backward_operation=exp_backward
        )

        exp_x = np.exp(x)
        self.autograd_engine.add_operation(
            inputs=[x],
            output=exp_x,
            gradients_to_update=[None],  
            backward_operation=exp_backward
        )

        exp_x_minus_exp_neg_x = exp_x - exp_neg_x
        self.autograd_engine.add_operation(
            inputs=[exp_x, exp_neg_x],
            output=exp_x_minus_exp_neg_x,
            gradients_to_update=[None, None],  
            backward_operation=sub_backward
        )

        exp_x_add_exp_neg_x = exp_x + exp_neg_x
        self.autograd_engine.add_operation(
            inputs=[exp_x, exp_neg_x],
            output=exp_x_add_exp_neg_x,
            gradients_to_update=[None, None],  
            backward_operation=add_backward
        )

        tanh = exp_x_minus_exp_neg_x / exp_x_add_exp_neg_x
        self.autograd_engine.add_operation(
            inputs=[exp_x_minus_exp_neg_x, exp_x_add_exp_neg_x],
            output=tanh,
            gradients_to_update=[None, None],  
            backward_operation=div_backward
        )

        self.state = tanh


        return self.state


class ReLU(Activation):
    """
    ReLU activation.
    Feel free to enumerate primitive operations here (you may find it helpful).
    Feel free to refer back to your implementation from part 1 of the HW for equations.
    """
    def __init__(self, autograd_engine):
        super(ReLU, self).__init__(autograd_engine)

    def forward(self, x):

        # TODO Compute forward with primitive operations
        # TODO Add operations to the autograd engine as you go
        
        # Step 1: Apply ReLU (element-wise max(0, x))
        zero = np.zeros_like(x)
        relu_output = np.maximum(zero, x)

        # Step 2: Add the operation to the autograd engine
        self.autograd_engine.add_operation(
            inputs=[x],
            output=relu_output,
            gradients_to_update=[None],  
            backward_operation=max_backward
        )

        # Save the result in self.state for later use
        self.state = relu_output

        # Return the result
        return relu_output
