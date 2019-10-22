
import numpy as np
import random
from keras.models import model_from_json
#import collections as coll
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
import params



        #self.brain              = self._build_model()
def _build_model():
   #Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Dense(30, input_dim=params.state_size, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(params.action_size, activation='linear'))
    model.compile(loss='mse', optimizer=adam(lr=params.alpha))

def act(state):
    if np.random.rand() <= params.exploration_rate:
        return random.randrange(params.action_size)
    act_values = (_build_model()).predict(state)
    return np.argmax(act_values[0])
    
def remember(state, action, reward, next_state):
    params.memory.append((state, action, reward, next_state))
        
        
def replay(params.sample_batch_size):
        if len(params.memory) < params.sample_batch_size:
            return
        params.sample_batch = random.sample(params.memory, params.sample_batch_size)
        for state, action, reward, next_state in params.sample_batch:
            #target = reward
            target = reward + params.gamma * np.amax((_build_model()).predict(next_state)[0])
            target_f = (_build_model()).predict(state)
            target_f[0][action] = target
            (_build_model()).fit(state, target_f, epochs=10, verbose=2)
        if params.exploration_rate > params.exploration_min:
            params.exploration_rate *= params.exploration_decay
            
def execute(next_state, reward):

        #state = np.reshape(state, (12,))
     
            
        #self.next_state = np.reshape(next_state, (12, ))
        
        profit = reward
            
        remember(params.state, params.action, profit, next_state)
            
        params.state = next_state
            
        params.action = act(params.state)
     
        replay(params.sample_batch_size)
     
        return params.action


            
           
    
            

