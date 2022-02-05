from ExperienceReplay import REPLAY_BUFFER
from baseclass import tf_PORTFOLIO
from tensorflow import keras
import tensorflow as tf

C  = 0.01


class TD3(tf_PORTFOLIO):
    def __init__(self, N, num_features,  maxsize, actor_lr, critic_lr, gamma, tau):
        super().__init__(N)
        self.Buffer = REPLAY_BUFFER(maxsize)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate = actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate = critic_lr)
        self.gamma = gamma
        self.tau = tau
        self.num_features = num_features
        self.prev_allocation = tf.Variable(tf.zeros_like(self.pf_allocation, dtype=tf.float32))
        self.prev_value = tf.Variable(1, dtype = tf.float32)
        self.create_actor()
        self.create_critic()
        self.reset()
        #self.noise = tf.Variable(tf.zeros_like(self.pf_allocation, dtype=tf.float32))


    @tf.function
    def reset(self):
        self.reset_portfolio()

    @tf.function
    def rebalance(self, state):
        #self.noise.assign(tf.random.normal(shape = [self.N + 1], mean = tf.zeros_like(self.pf_allocation, dtype=tf.float32), stddev = 0.2) ** 2)
        self.prev_allocation.assign(self.pf_allocation)
        self.new_allocation.assign(self.actor(state))
        self.rebalance_portfolio(self.new_allocation)


    @tf.function
    def step(self, state, returns):
        self.prev_value.assign(self.pf_value)
        self.step_portfolio(returns)
        self.rebalance(state)
        return self.pf_allocation, self.prev_allocation, self.gamma * tf.math.log(self.pf_value/self.prev_value) 


    @tf.function
    def train_critic(self, S, W, A, R, S_, W_):
        with tf.GradientTape() as tape:
            A_target = self.actor_target([S_, W_])
            y_target = R + self.gamma * tf.math.minimum(self.critic_target_1([S_, W_, A_target]), self.critic_target_2([S_, W_, A_target]))
            y_predict = self.critic([S, W, A])
            critic_loss = tf.math.reduce_mean(tf.math.square(y_target - y_predict))

        critic_grads = tape.gradient(critic_loss,self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads,self.critic.trainable_variables))

    @tf.function
    def train_actor(self, W, S, A, R, W_, S_):
        with tf.GradientTape() as tape:
            Actions = self.actor([S, W])
            Q = self.critic([S, W, Actions])
            actor_loss = -tf.math.reduce_mean(Q)
            
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

    def create_actor(self):
        S =  keras.Input(shape = (None, self.N, self.num_features))
        W =  keras.Input(shape = (None, self.N, 1))

        y = keras.layers.Concatenate()([S, W])
        y = keras.layers.Dense(50, activation='linear')(S)
        y = keras.layers.Dense(500, activation='relu')(y)
        y = keras.layers.Dense(500, activation='relu')(y)
        y = keras.layers.Dense(50, activation='sigmoid')(y)
        y = keras.layers.Dense(1, activation='linear')(y)
        y = keras.layers.Flatten()(y)
        output = y / tf.math.reduce_sum(y)
        
        self.actor = keras.Model(inputs = [S, W], outputs = output)
        self.actor_target = keras.Model(inputs = [S, W], outputs = output)
        self.actor_target.set_weights(self.actor.get_weights())

    def create_critic(self):
        S =  keras.Input(shape = (None, self.N, self.num_features))
        W =  keras.Input(shape = (None, self.N, 1))
        A =  keras.Input(shape = (None, self.N, 1))

        y = keras.layers.Concatenate()([S, W])
        y = keras.layers.Dense(50, activation='linear')(y)
        y = keras.layers.Dense(500, activation='relu')(y)
        y = keras.layers.Dense(500, activation='relu')(y)
        y = keras.layers.Dense(50, activation='sigmoid')(y)
        y = keras.layers.Dense(1, activation='linear')(y)
        output = tf.math.reduce_sum(y) *  (1 - C * tf.norm(W - A, ord=1))

        self.critic = keras.Model(inputs = [S, W, A], outputs = output)
        self.critic_target_1 = keras.Model(inputs = [S, W, A], outputs = output)
        self.critic_target_2 = keras.Model(inputs = [S, W, A], outputs = output)
        self.critic_target_1.set_weights(self.critic.get_weights())
        self.critic_target_2.set_weights(self.critic.get_weights())
    
    @tf.function
    def Update_target_weights(self):
        for (a, b) in zip(self.critic_target_1.variables, self.critic.variables):
            a.assign(a * (1 - self.tau) + b * self.tau)
        for (a, b) in zip(self.critic_target_2.variables, self.critic.variables):
            a.assign(a * (1 - self.tau) + b * self.tau)
        for (a, b) in zip(self.actor_target.variables, self.actor.variables):
            a.assign(a * (1 - self.tau) + b * self.tau)


X = TD3(2, 2, 10000, 0.1, 0.1 , 0.99, 0.95)
import numpy as np
S = np.zeros((X.N, X.num_features))

W =  np.zeros((X.N, 1))

A =  np.zeros((X.N, 1))

#print(X.actor([S,W]))
print(X.critic([S, A, A]))
