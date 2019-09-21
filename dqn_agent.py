#!/usr/bin/python3
import numpy as np
import random
import collections as coll
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam

class DQNAgent():
    def __init__(self):
        self.state_size         = 12
        self.action_size        = 7
        self.memory             = coll.deque(maxlen=3000)
        self.learning_rate      = 0.5
        self.gamma              = 0.7
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.state              = 0
        self.action             = 1
        self.brain              = self._build_model()

    def _build_model(self):
        #Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=adam(lr=self.learning_rate))

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])
    
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        
    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state in sample_batch:
            #target = reward
            target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=2)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
            
    def execute(self, next_state, reward):

        #state = np.reshape(state, (12,))
     
            
        #self.next_state = np.reshape(next_state, (12, ))
        
        profit = reward
            
        self.remember(self.state, self.action, profit, next_state)
            
        self.state = next_state
            
        self.action = self.act(self.state)
     
        self.replay(32)
     
        return self.action
 
 
if __name__ == "__main__":
     
    agent = DQNAgent()
     
    result = agent.execute(5, 117.456)
     
    print(result)
     
     
            
           
    
            

