"""
This module contains classes of fitness mapping function.
"""
import torch



class ProbMapping:
    """
    A class for adapting to a unified interface, supporting multiple mapping functions.
    
    Supported mapping methods：
        - exp: exp(-x / T)
        - rational: 1 / (1 + a * x)
        - sigmoid: sigmoid(x / T)
        - identity: x
    """

    def __init__(self, method='exp', T=1.0, a=1.0):
        """
        :param method: str,  ['exp', 'rational', 'sigmoid', 'identity']
        :param T: float,  T
        :param a: float, rational 
        """
        self.method = method
        self.T = T
        self.a = a


    def __call__(self, x):
        """
        Call the interface and execute the mapping according to the set method.
        """
        return self.forward(x)

    def forward(self, x):
        """
        :param x: torch.Tensor, input fitness 
        :return: torch.Tensor, result
        """
        if self.method == 'exp':
            return torch.exp(x)
        elif self.method == 'rational':
            return 1.0 / (1.0 + x)
        elif self.method == 'sigmoid':
            return torch.sigmoid(x)
        elif self.method == 'identity':
            return x
        else:
            raise ValueError(f"Unsupported method: {self.method}")








