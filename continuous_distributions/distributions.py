#%%
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#%%
import random
import matplotlib.pyplot as plt
import numpy as np
import collections

#%%
class ProbDist:
    def __init__(self, dim):
        self.dim = dim
        if self.dim == 1:
            pass
        else:
            raise NotImplementedError

#%%
class UnivPD(ProbDist):
    def __init__(self, dim = 1):
        super().__init__(dim)

class MultivPD(ProbDist):
    def __init__(self, dim):
        super().__init__(dim)

#a = UnivPD()
#b = MultivPD(dim = 2) #same as univProbDist, will raise Error if dimensionality != 1




