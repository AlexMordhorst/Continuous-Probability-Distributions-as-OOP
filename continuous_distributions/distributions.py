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

#%%
class ContinuousDistribution(UnivPD):
    def __init__(self, **parameters):
        for k, v in parameters.items():
            setattr(self, k , v)

    def densityFunction(self, x): #warum x als parameter und nicht k?
        if k >= self.theta1 and k <= self.theta2:
            return 1/(self.theta1-self.theta2)
        else:
            return 0

    def vectorizedDensity(self):
        return np.vectorize(self.densityFunction) #vectorize function on what?

    def integrate(self, f, a, b, steps = 50, plot = False):
        step_size = (b-a)/steps
        x = np.linspace(a, b, steps)
        y = f(x)
        integral = np.sum(y*step_size)
        if plot is True:
            self.plotFunction(x,y)
        return integral

    def plotFunction(self, x, y):
        plt.plot(x,y, color = "r")
        plt.fill_between(x,y)
        plt.xlim([x[0],x[-1]])
        plt.ylim([0, max(y)+1])#warum ylim nicht 1? bzw da auc von density immer 1, macht >1 hier wenig sinn
        plt.show()

    def probability(self, a, b):
        probability = self.integrate(self.vectorizedDensity(), a, b)
        return probability

    def expectedValue(self):
        expected_value = None
        return expected_value

    def variance(self):
        variance = None
        return variance

#%%
class UniformDistribution(ContinuousDistribution):
    def __init__(self, theta1, theta2):
        self.theta1 = theta1
        self.theta2 = theta2

    def densityFunction(self, k): #hier nunr k als parameter
        if k >= self.theta1 and k <= self.theta2:
            return 1/(self.theta2-self.theta1)
        else:
            return 0

    def probability(self, a, b):
        probability = self.integrate(self.vectorizedDensity(), a, b, steps = 50, plot= True)
        return probability

    def expectedValue(self):
        return (self.theta1 + self.theta2)/2

    def variance(self):
        return ((self.theta2 - self.theta1)**2)/12

#%%
a = UniformDistribution(theta1=0, theta2 = 30)
a.probability(25,30)
a.expectedValue()
a.variance()