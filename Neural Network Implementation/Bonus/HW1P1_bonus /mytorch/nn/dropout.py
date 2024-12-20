# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):

        if train:
            # TODO: Generate mask and apply to x
            self.mask = np.random.binomial(1, 1 - self.p, size=x.shape)

            x = self.mask * x / (1 - self.p)

            return x
            
        else:
            # TODO: Return x as is

            return x
		
    def backward(self, delta):
        # TODO: Multiply mask with delta and return

        self.mask = self.mask * delta

        return self.mask