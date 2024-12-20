import numpy as np
import torch

from mytorch import autograd_engine
import mytorch.nn as nn
from helpers import *
from mytorch.functional_hw1 import *
    
# Activation Layer test - for Sigmoid, Tanh and ReLU

def test_identity_forward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))  # just use batch size 1 since broadcasting is not written yet
    l1_out = l1(x)
    test_act = nn.Identity(autograd)
    a1_out = test_act(l1_out)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))  # note transpose here, probably should standardize
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Identity()
    torch_a1_out = torch_act(torch_l1_out)

    compare_np_torch(a1_out, torch_a1_out)
    
    return True


def test_identity_backward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.Identity(autograd)
    a1_out = test_act(l1_out)
    autograd.backward(1)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Identity()
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()

    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)
    
    return True


def test_sigmoid_forward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.Sigmoid(autograd)
    a1_out = test_act(l1_out)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Sigmoid()
    torch_a1_out = torch_act(torch_l1_out)

    compare_np_torch(a1_out, torch_a1_out)

    return True


def test_sigmoid_backward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))

    print("\nInitial Input (X):", x)
    print("\nInitial Weights (W):", l1.W)
    print("Initial Biases (b):", l1.b)


    l1_out = l1(x)
    test_act = nn.Sigmoid(autograd)
    a1_out = test_act(l1_out)

    print("\nForward pass result:", a1_out)


    autograd.backward(1)

    print("\nCustom Autograd Gradients:")
    print("Weight gradient (dW):", l1.dW)
    print("Bias gradient (db):", l1.db)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Sigmoid()
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()
    
    print("\nTorch Gradients:")
    print("Weight gradient (Torch dW):", torch_l1.weight.grad)
    print("Bias gradient (Torch db):", torch_l1.bias.grad)

    print("Checking for equality:")
    print("Weights match:", np.allclose(l1.dW, torch_l1.weight.grad))
    print("Biases match:", np.allclose(l1.db.squeeze(), torch_l1.bias.grad))

    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)
    
    return True


def test_tanh_forward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.Tanh(autograd)
    a1_out = test_act(l1_out)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Tanh()
    torch_a1_out = torch_act(torch_l1_out)

    compare_np_torch(a1_out, torch_a1_out)
    return True


def test_tanh_backward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.Tanh(autograd)
    a1_out = test_act(l1_out)
    autograd.backward(1)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Tanh()
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()

    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)
    
    return True

def test_relu_forward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    # print(l1_out.shape)
    test_act = nn.ReLU(autograd)
    a1_out = test_act(l1_out)
    # print(a1_out.shape)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    # print(torch_l1_out.shape)
    torch_act = torch.nn.ReLU()
    torch_a1_out = torch_act(torch_l1_out)
    # print(torch_a1_out.shape)

    compare_np_torch(a1_out, torch_a1_out)
    
    return True


def test_relu_backward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.ReLU(autograd)
    a1_out = test_act(l1_out)
    autograd.backward(1)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.ReLU()
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()
    
    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)
    
    return True
