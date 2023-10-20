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

    def pdf(self, k): #warum x als parameter und nicht k?
        if k >= self.theta1 and k <= self.theta2:
            return 1/(self.theta1-self.theta2)
        else:
            return 0

    def vectorizedDensity(self):
        return np.vectorize(self.pdf) #vectorize function on what?

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

    def cdf(self, a, b):
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

    def pdf(self, k): #hier nunr k als parameter
        if k >= self.theta1 and k <= self.theta2:
            return 1/(self.theta2-self.theta1)
        else:
            return 0

    def cdf(self, a, b):
        probability = self.integrate(self.vectorizedDensity(), a, b, steps = 50, plot= True)
        return probability

    def expectedValue(self):
        return (self.theta1 + self.theta2)/2

    def variance(self):
        return ((self.theta2 - self.theta1)**2)/12

#%%
#a = UniformDistribution(theta1=0, theta2 = 30)
#a.cdf(25,30)
#a.expectedValue()
#a.variance()

#%%
class NormalDistribution(ContinuousDistribution):
    """
    
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, k):
        exp = np.exp(-(((k - self.mu)**2)/(2*self.sigma**2)))
        return ((1/(self.sigma*np.sqrt(2*np.pi)))*exp)

    def cdf(self, a, b):
        if self.sigma > 0:
            probability = self.integrate(self.vectorizedDensity(), a, b, steps=100, plot = True)
            return probability
        else:
            pass

    def expectedValue(self):
        return self.mu

    def variance(self):
        return self.sigma**2

#%%
#dist = NormalDistribution(mu = 75, sigma = 10)
#dist.cdf(80,90)
#dist.expectedValue()
#dist.variance()

#%%
class TDistribution(ContinuousDistribution):
    """

    """
    def __init__(self, df):
        self.df = df
    def pdf(self, k):
        factor = np.math.gamma(self.df + 1 / 2)/np.sqrt(self.df * np.pi)*np.math.gamma(self.df/2)
        return factor * (1 + k**2/self.df)**-(self.df+1/2)

    def cdf(self, a, b):
        probability = self.integrate(self.vectorizedDensity(), a, b, steps=100, plot = True)
        return probability

    def expectedValue(self):
        return 0

    def variance(self):
        if self.df > 2:
            return self.df/(self.df - 2)
        elif self.df == 2:
            return np.math.inf
        else:
            return ValueError

#%%
#tdist = TDistribution(df= 30)
#tdist.cdf(0.8,0.9)