import os
from ptycho.pmace import pmace_recon


class PMACE():
    """This class is a decorator that can be used to prepare a function before it is called.

    Args:
        func (function): The function to be decorated.
        *args: Positional arguments to be passed to the decorated function.
        **kwargs: Keyword arguments to be passed to the decorated function.
    """
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs   
        
    def __call__(self):
        def wrapper(*args, **kwargs):
            print('Preparing function ...')
            return_val = self.func(*args, **kwargs)
            return return_val
        
        return wrapper(*self.args, **self.kwargs)
    
