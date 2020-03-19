import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

import random

import params

def make_samples(samples):
    array = np.array(samples)
	
    current_states = np.stack(array[:,0]).reshape((array.shape[0],-1))
    actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
    rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
    new_states = np.stack(array[:,3]).reshape((array.shape[0],-1))
	
    return current_states, actions, rewards, new_states
	

sess = tf.Session()
K.set_session(sess)

# Please set these values in params.py 
'''
learning_rate = 0.0001
epsilon = .9
epsilon_decay = .99995
gamma = .8
tau   = .01

'''
	

actor_state_input, actor_model = create_actor_model()
_, target_actor_model = create_actor_model()

actor_critic_grad = tf.placeholder(tf.float32, [None, params.action_size]) # where we will feed de/dC (from critic)

actor_model_weights = actor_model.trainable_weights
actor_grads = tf.gradients(actor_model.output, actor_model_weights, -actor_critic_grad) # dC/dA (from actor)
grads = zip(actor_grads, actor_model_weights)
optimize = tf.train.AdamOptimizer(params.learning_rate).apply_gradients(grads)


critic_state_input, critic_action_input, critic_model = create_critic_model()
_, _,  target_critic_model = create_critic_model()

critic_grads = tf.gradients(critic_model.output, critic_action_input) # where we calcaulte de/dC for feeding above

        
sess.run(tf.initialize_all_variables())



def create_actor_model():
    state_input = Input(shape=(params.state_size,))
    h1 = Dense(100, activation='relu')(state_input)
    h2 = Dense(50, activation='relu')(h1)
    h3 = Dense(20, activation='relu')(h2)
    output = Dense(params.action_size, activation='tanh')(h3)

    model = Model(input=state_input, output=output)
    adam  = Adam(lr=0.0001)
    model.compile(loss="mse", optimizer=adam)
    return state_input, model

def create_critic_model():
    state_input = Input(shape=(params.state_size,))
    state_h1 = Dense(200, activation='relu')(state_input)
    state_h2 = Dense(100)(state_h1)

    action_input = Input(shape=(params.action_size,))
    action_h1    = Dense(50)(action_input)

    merged    = Concatenate()([state_h2, action_h1])
    merged_h1 = Dense(20, activation='relu')(merged)
    output = Dense(1, activation='linear')(merged_h1)
    model  = Model(input=[state_input,action_input], output=output)

    adam  = Adam(lr=0.0001)
    model.compile(loss="mse", optimizer=adam)
    return state_input, action_input, model



def remember(cur_state, action, reward, new_state):
    params.memory.append([cur_state, action, reward, new_state])

def _train_actor(samples):
		
    cur_states, actions, rewards, new_states =  make_samples(samples)
    predicted_actions = actor_model.predict(cur_states)
    grads = sess.run(critic_grads, feed_dict={
	critic_state_input:  cur_states,
	critic_action_input: predicted_actions
    })[0]

    sess.run(optimize, feed_dict={
	actor_state_input: cur_states,
	actor_critic_grad: grads
    })

def _train_critic(samples):
   

    cur_states, actions, rewards, new_states = make_samples(samples)
    target_actions = target_actor_model.predict(new_states)
    future_rewards = target_critic_model.predict([new_states, target_actions])
		
    rewards += params.gamma * future_rewards
		
    evaluation = critic_model.fit([cur_states, actions], rewards, verbose=1)
    
def train():
		
    if len(params.memory) < params.batch_size:
	return

    rewards = []
    samples = random.sample(params.memory, params.batch_size)
    _train_critic(samples)
    _train_actor(samples)


    

def _update_actor_target():
    actor_model_weights  = actor_model.get_weights()
    actor_target_weights = target_actor_model.get_weights()
		
    for i in range(len(actor_target_weights)):
	actor_target_weights[i] = actor_model_weights[i]*params.tau + actor_target_weights[i]*(1-params.tau)
    target_actor_model.set_weights(actor_target_weights)

def _update_critic_target():
    critic_model_weights  = critic_model.get_weights()
    critic_target_weights = target_critic_model.get_weights()
		
    for i in range(len(critic_target_weights)):
	critic_target_weights[i] = critic_model_weights[i]*params.tau + critic_target_weights[i]*(1-params.tau)
    target_critic_model.set_weights(critic_target_weights)

def update_target():
    _update_actor_target()
    _update_critic_target()



def act(state):
    if np.random.rand() <= params.epsilon:
        return random.uniform(0.5, 10.0)
        
    act_values = actor_model.predict(state)
    return act_values[0][0]


def execute(state, reward):

    agent_next_state = [0]*params.state_size

    agent_next_state[state] = 1

    agent_next_state = np.array(agent_next_state)

    agent_next_state = np.reshape(agent_next_state, [1, params.state_size])

    remember(params.agent_current_state, params.action, reward, agent_next_state)

    params.agent_current_state = agent_next_state

    params.action = act(params.agent_current_state)

   #Please write the logic over here. Want to train/update the networks every 120 hours. Should be somethign like this.

   # if(no. of days % 120 == 0)
   #{
   #    train()
   #    update_target()
    #}
       

    return params.action

