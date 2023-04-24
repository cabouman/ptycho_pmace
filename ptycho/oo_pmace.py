import os
from ptycho.pmace import pmace_recon


class PMACE():
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
    
