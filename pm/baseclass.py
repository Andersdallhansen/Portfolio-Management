from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import tensorflow as tf

C = 0.01


class PORTFOLIO(ABC):
    @abstractmethod
    def __init__(self, N, *args):
        pass
    
    def reset_portfolio(self):
        self.pf_value = 1
        self.pf_allocation = np.array([1.] + [0.] * self.N).astype('float32')

    def rebalance_portfolio(self, new_allocation):
        self.pf_value = self.pf_value * (1- C * np.linalg.norm(self.pf_allocation[1:] - new_allocation[1:], ord=1))
        self.pf_allocation = new_allocation

    def step_portfolio(self, returns):
        pf_return = np.dot(returns, self.pf_allocation)
        self.pf_value = self.pf_value * pf_return
        self.pf_allocation = self.pf_allocation * returns / pf_return

    @abstractmethod
    def reset(self, *args):
        pass

    @abstractmethod
    def rebalance(self, *args):
        pass

    @abstractmethod
    def step(self, *args):
        pass


class tf_PORTFOLIO(ABC):
    def __init__(self, N):
        self.N = N
        self.pf_value = tf.Variable(1, dtype=tf.float32)
        self.pf_allocation = tf.Variable([1.] + [0.] * self.N, dtype=tf.float32)
    
    @tf.function
    def reset_portfolio(self):
        self.pf_value.assign(1)
        self.pf_allocation.assign([1.] + [0.] * self.N)
        #self.pf_allocation.assign(tf.zeros_like(self.pf_allocation, dtype=tf.float32))
        #self.pf_allocation[0].assign(1)

    @tf.function
    def rebalance_portfolio(self, new_allocation):
        self.pf_value.assign(self.pf_value * (1 - C * tf.norm(self.pf_allocation[1:] - new_allocation[1:], ord=1)))
        self.pf_allocation.assign(new_allocation)

    @tf.function
    def step_portfolio(self, returns):
        pf_return = tf.tensordot(returns, self.pf_allocation)
        self.pf_value.assign(self.pf_value * pf_return)
        self.pf_allocation.assign(self.pf_allocation * returns / pf_return)

    @abstractmethod
    @tf.function
    def reset(self, *args):
        pass

    @abstractmethod
    @tf.function
    def rebalance(self, *args):
        pass

    @abstractmethod
    @tf.function
    def step(self, *args):
        pass