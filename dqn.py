from keras.models import Sequential
from keras.layers import Dense
import random
import numpy as np
import params
        
model = Sequential()
model.add(Dense(24, input_dim=params.state_size, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(params.action_size, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=params.alpha))

    
def remember(state, action, reward, next_state):
    params.memory.append((state, action, reward, next_state))
        
def act(state):
    if np.random.rand() <= params.epsilon:
        return random.randrange(params.action_size)
        
    act_values = model.predict(state)
    return np.argmax(act_values[0])
    
def replay():

    if (len(params.memory) < params.batch_size):
        return

    minibatch = random.sample(params.memory, params.batch_size)
    for state, action, reward, next_state in minibatch:
        #target = reward
        target = reward + params.gamma * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=2)

    if params.epsilon > params.epsilon_min:
        params.epsilon *= params.epsilon_decay


def execute(state, reward):

    agent_next_state = [0]*params.state_size

    agent_next_state[state] = 1

    agent_next_state = np.array(agent_state)

    remember(params.agent_current_state, params.action, reward, agent_next_state)

    params.agent_current_state = agent_next_state

    params.action = act(params.agent_current_state)

    replay()

    return params.action

        

        

        
        

        
