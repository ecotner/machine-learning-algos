# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:39:27 2017

Implementation of a Q-network which plays Pong (Atari game).

Description:
This project uses a neural network as a function approximator in order to estimate the function Q(s,a) associated with the expected future reward for choosing action a while in state s. Here, s is a sequence of images representing the last 4 frames of a game of Pong, and a is one of {do nothing, up, down}. The input to the network is simply the frame data (representing s), and the output is a 3-D vector Q* representing the expected values of Q for the 3 choices of a. We use an epsilon-greedy algorithm to select the action (take random a with prob epsilon, otherwise argmax(Q)), use that to update the state s -> s', then calculate the expected Q using the Bellman equation Q(s,a) = r + \gamma*argmax(Q*(s',a')), then compute the loss via L = (1/2)*(Q*(s,a) - Q(s,a))^2 (and perform backprop appropriately). This causes the Q-network to learn to approximate Q(s,a), which can then be used to take the optimal action in a given situation.

To do:
    - Finish writing model_run() function
    - Rewrite action selection algorithm so that (if not random) the action is chosen from a distribution of the form softmax(Q*)
    - Do proper vectorization of batches
    - Write a function to save training data between runs (not the weights, TF already takes care of that)
    - Write a function to automatically handle plotting the training data
    - Write functions that:
        1) Collect a series of frames into a static validation set before training start
        2) Run validation data periodically on static validation set and plot <Q>
    - Figure out why the training loss is spiking, and why it spikes to the same value all the time.

Things I (think I) learned:
    1) The discount factor \gamma should be chosen so that \gamma^n = 1/e, where n is the average number of time steps per reward. This makes it so that the characteristic decay timescale of the future rewards is roughly the time between rewards, ensuring the most recent reward is always relevant. (Perhaps I should make \gamma adaptive, which would just remove it as a hyperparameter altogether).
    2) The initialization of the weights in the final layer are very important as well. You want to initialize so that the output is very close to what you would expect to get for losing all your games almost immediately, i.e. if it takes n frames to lose a game quickly, and you choose \gamma^n = 1/e as prescribed above, your expected Q on average is about <Q> ~ \gamma^n r (= -1/e ~ 0.37 in this case). You can do this by making b = <Q> >> rms(W) in the output layer. If not initialized this way, you should see your batch mean <Q> move toward this value, causing large spikes in the training loss along the way. Then as the network improves, it should ideally increase.

@author: Eric Cotner
"""


import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
import time

''' ===================== SET UP PONG ENVIRONMENT ======================= '''
env = gym.make('Pong-v0')
action_space = env.action_space
# 0,1 = do nothing, 2,4 = up, 3,5 = down (just use 1,2,3)
# These commands actually impart a momentum to the paddle in the specified direction! So you can do 2,2,0,0,0 and actually coast a significant part of the way to the top of the stage!
observation_space = env.observation_space
# Box of shape (210,16,3) = (height,width,color channels)

''' ===================== DATA PROCESSING AND HANDLING ===================== '''

#raw_input_shape = observation_space.shape

def process_raw_frame(input_data, color=False, crop=True, ds_factor=1):
    ''' Takes the raw image data, crops it down so that only the play space (160x160 box) is visible, takes a single slice of the RGB color channels (channel 2 seems to have the most contrast), and then downsamples it.
    Args:
        input_data: The raw data, in a 210x160x3 array.
        ds_factor: the factor by which to downsample. Must be integer i >= 1 which is a divisor of 160 (1,2,4,5,8,10,16,20,32,40,80,160).
        crop: Whether or not to crop to 160x160 array
    Output:
        processed_data: An array of the processed output data with shape (1,height,width,color).
        '''
    if color == False:
        processed_data = input_data[:,:,2]
        if crop == True:
            processed_data = processed_data[34:194,:]
        m, n = processed_data.shape
        if ds_factor != 1:
            processed_data_temp = np.zeros((m//ds_factor,n//ds_factor))
            for i in range(m//ds_factor):
                for j in range(n//ds_factor):
                    processed_data_temp[i,j] = np.max(processed_data[ds_factor*i:ds_factor*(i+1),ds_factor*j:ds_factor*(j+1)])
            processed_data = processed_data_temp
        out_shape = (1,m,n,1)
    else:
        processed_data = input_data[:,:,:]
        if crop == True:
            processed_data = processed_data[34:194,:,:]
        m, n, _ = processed_data.shape
        if ds_factor != 1:
            processed_data_temp = np.zeros((m//ds_factor,n//ds_factor,3))
            for i in range(m//ds_factor):
                for j in range(n//ds_factor):
                    for k in range(3):
                        processed_data_temp[i,j,k] = np.max(processed_data[ds_factor*i:ds_factor*(i+1),ds_factor*j:ds_factor*(j+1),k])
            processed_data = processed_data_temp
        out_shape = (1,m,n,3)
    return processed_data.reshape(out_shape)

def add_to_frame_list(frame_list, frames_to_add, max_frames_to_keep=5):
    ''' Adds a set of frames to the list frame_list. Modifies the frame_list IN PLACE, so doesn't return anything. Assumes frames_to_add is a LIST.'''
    assert type(frames_to_add) == list, 'frames_to_add must be a list'
    if len(frame_list) + len(frames_to_add) <= max_frames_to_keep:
        for frame in frames_to_add:
            frame_list.append(frame)
    else:
        for frame in frames_to_add:
            frame_list.append(frame)
        diff = len(frame_list) - max_frames_to_keep
        for i in range(diff):
            frame_list.pop(0)

#def sample_from_frame_list(frame_list, nsamples=5, sample_len=4):
#    ''' Takes nsamples random samples of sample_len adjacent frames from frame_list and returns a batch of stacked frames of dimension (nsamples,160,160,sample_len). '''
#    batch = np.zeros((nsamples,160,160,sample_len))
#    perm_idx = np.random.permutation(np.arange(sample_len-1, len(frame_list)))[:nsamples]
#    for sample_idx in range(nsamples):
#        for frame_idx in range(sample_len):
#            batch[sample_idx,:,:,frame_idx] = frame_list[perm_idx[sample_idx]-frame_idx][0,:,:,0]
#    return batch

def stack_frames(frame_list):
    ''' Stacks a list of frames of dimension (1,160,160,1) into a single array of dimension (1,160,160,len(frame_list)). '''
    stack = np.zeros((1,160,160,len(frame_list)))
    for frame_idx in range(len(frame_list)):
        stack[0,:,:,frame_idx] = frame_list[-(1+frame_idx)][0,:,:,0]
    return stack

def add_to_replay_memory(replay_memory, experience, max_exp_to_keep=600):
    ''' Adds the experience tuple (frames, action, reward, next_frames, done) to the replay_memory list. Modifies replay_memory IN PLACE, so doesn't return anything. '''
    assert type(experience) == tuple, 'frames_to_add must be a tuple'
    if len(replay_memory) + 1 <= max_exp_to_keep:
        replay_memory.append(experience)
    else:
        replay_memory.append(experience)
        replay_memory.pop(0)

def sample_from_replay_memory(replay_memory, nsamples=5, include_most_recent=True):
    ''' Takes nsamples random samples from replay_memory and returns 'samples', a list of tuples, each of which is an experience containing (frames, action, reward, next_frames, done). '''
    samples = []
    if nsamples >= len(replay_memory):
        samples = replay_memory[:]
    else:
        if include_most_recent == True:
            samples.append(replay_memory[-1])
            nsamples += -1
        perm_idx = np.random.permutation(len(replay_memory))[:nsamples]
        for sample_idx in range(nsamples):
#            print(perm_idx)
#            print(sample_idx)
#            print(nsamples)
            samples.append(replay_memory[perm_idx[sample_idx]])
    return samples

"""
def sample_from_replay_memory_2(replay_memory, nsamples=5, include_most_recent=True):
    ''' Takes nsamples random samples from replay_memory and returns 'samples', a list of tuples, each of which is an experience containing (frames, action, reward, next_frames, done). '''
    samples = []
    # Create list of sampled experiences 'samples'
    if nsamples >= len(replay_memory):
        samples = replay_memory[:]
    else:
        if include_most_recent == True:
            samples.append(replay_memory[-1])
            nsamples += -1
        perm_idx = np.random.permutation(len(replay_memory))[:nsamples]
        for sample_idx in range(nsamples):
#            print(perm_idx)
#            print(sample_idx)
#            print(nsamples)
            samples.append(replay_memory[perm_idx[sample_idx]])
    # Extract sequences, actions, and rewards
"""

''' =================== SET UP COMPUTATIONAL GRAPH ===================== '''

def initialize_graph(in_height=160, in_width=160, color=1, nframes=4):
    '''
    Initializes the computational graph for the neural network. The arguments are pretty self-explanatory, but should not be modified since I'm not sure how the padding on the convolutional layers works exactly, so the number of nodes in the dense layer after the 2nd convolutional layer is hard-coded based on the default arguments; this number will need to be changed if the input dimensions are changed. The output is 'G', a tensorflow Graph object containing the computational graph. All the tensor declarations are local to this function which avoids polluting the global namespace, but can still be accessed using the Graph.get_tensor_by_name() method. (Need to make sure all tensors of interest have names!)
    '''
    
    tf.reset_default_graph()
    G = tf.Graph()
    
    with G.as_default():
        # Reset default graph
#        tf.reset_default_graph()
        xavier_init = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        
        # Input layer
        X = tf.placeholder(dtype=tf.float32, shape=[None,in_height,in_width,nframes*color], name='X')
        
        # Convolutional and pooling layer(s)
        W1 = tf.get_variable(shape=[10,10,nframes*color,32], initializer=xavier_init, name='W1')
        b1 = tf.Variable(0.01*np.ones((32,)), dtype=tf.float32, name='b1')
        conv1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,2,2,1], padding='SAME')+b1, name='conv1')
        # conv1 has dimension (batch_size,(in_height-10)/2+1,(in_width-10)/2+1,32), e.g. (batch_size,76,76,32) for default values
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
        # pool1 has dimension (batch_size,((in_height-10)/2+1)/2,((in_width-10)/2+1)/2,32), e.g. (batch_size,38,38,32) for default values
        
        W2 = tf.get_variable(shape=[4,4,32,64], initializer=xavier_init, name='W2')
        b2 = tf.Variable(0.01*np.ones((64,)), dtype=tf.float32, name='b2')
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W2, strides=[1,2,2,1], padding='SAME')+b2, name='conv2')
        # conv2 has dimension (batch_size,(((in_height-10)/2+1)/2-4)/2+1,(((in_width-10)/2+1)/2-4)/2+1,64), e.g. (batch_size,18,18,64) for default values
        pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME', name='pool2')
        # pool2 has dimension (batch_size,((((in_height-10)/2+1)/2-4)/2+1)/3,((((in_width-10)/2+1)/2-4)/2+1)/3,64), e.g. (batch_size,6,6,64) for default values
        
        # Dense layer(s)
#         dim_flat = tf.cast(tf.shape(pool2)[1]*tf.shape(pool2)[2]*tf.shape(pool2)[3], tf.int32, name='dim_flat')
        # Don't know why, but the output of pool2 has length 3136 rather than 6*6*64=2304. Probably due to padding...
        pool2_flat = tf.reshape(pool2, shape=[-1,3136], name='pool2_flat')
        W3 = tf.get_variable(shape=[3136,1024], initializer=xavier_init, name='W3')
        b3 = tf.Variable(0.01*np.ones((1024,)), dtype=tf.float32, name='b3')
        A3 = tf.nn.relu(tf.matmul(pool2_flat,W3)+b3, name='A3')
        
        # Output layer
        W4 = tf.get_variable(shape=[1024,3], initializer=xavier_init, name='W4')
        b4 = tf.Variable(np.zeros((3,)), dtype=tf.float32, name='b4')
        Q1 = tf.add(0.001*tf.matmul(A3,W4),b4, name='Q1')
        
        # Define predictions, loss, and accuracy metrics
        Q2 = tf.placeholder(tf.float32, shape=[], name='Q2')
        a = tf.placeholder(tf.int32, shape=[], name='a')
        loss = tf.multiply(0.5,tf.reduce_mean((Q1[:,a] - Q2)**2), name='loss')
         
    # Output Graph object
    return G

''' ================== SET UP OPTIMIZER AND TRAINING OPERATIONS =========== '''

def initialize_train_op(graph, lr):
    with graph.as_default():
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(graph.get_tensor_by_name('loss:0'))
    return train_op

''' ===================== TRAINING FUNCTIONS ============================= '''

def visualize_training(loss_list=[], Q_list=[], reward_list=[]):
    ''' Plots the loss and Q values over time '''
    if loss_list != []:
        plt.figure('Loss')
        plt.clf()
        plt.semilogy(loss_list, label='Training loss')
        plt.title('Training loss over batch')
        plt.xlabel('Time (arb. units)')
        plt.ylabel('Loss')
        plt.draw()
        plt.pause(1e-10)
    if Q_list != []:
        plt.figure('Q')
        plt.clf()
        plt.plot(Q_list, label='max(Q)')
        plt.title('Maximum value of mean expected Q over batch')
        plt.xlabel('Time (arb. units)')
        plt.ylabel('max(Q)')
        plt.draw()
        plt.pause(1e-10)
    if reward_list != []:
        plt.figure('Rewards')
        plt.clf()
        plt.plot(reward_list, label='Rewards')
        plt.title('Cumulative rewards over time')
        plt.xlabel('Time (arb. units)')
        plt.ylabel('Rewards')
        plt.draw()
        plt.pause(1e-10)

def model_train(lr, max_episodes, gamma, batch_size=5, epsilon0=0.1, plot_every_n_steps=100, n_steps_to_skip=2, save_every_n_episodes=5, recover_from_last_checkpoint=False, render=False):
    '''
    Training loop for the Q-network. Automatically initializes computational graph architecture, refreshes checkpoints if applicable, and runs the training loop over the required number of episodes. The training loop basically iterates through an episode frame by frame, at each step predicting the Q value for each action, given the previous 4 frames of the sequence. The action with the highest Q is picked (there is an epsilon chance of this decision being random), and then this action is used to step to the next frame. The Q values of the next frame are used to compute the expected Q, and then the Q-network weights are updated by backprop using a MSE loss.
    '''
    # Define some constants and set up random stuff
    nframes = 4
    height, width = (160,160)
    save_str = './checkpoints/Pong_2'
    reward_scale = 1
    plt.ion()
    # Set up Graph, Saver, Session, etc...
    G = initialize_graph()
    with G.as_default():
        saver = tf.train.Saver(var_list=None, max_to_keep=5)
        train_op = initialize_train_op(G, lr)
        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            # Recover from last checkpoint or save to checkpoint
            if recover_from_last_checkpoint == True:
                print('Recovering from last checkpoint...')
                saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
#            else:
#                print('Saving initial checkpoint...')
#                saver.save(sess, save_str)
            # Set up initial frame_list and replay_memory
            frame_list = []
            replay_memory = []
            # Make dictionary of tensors in graph needed for computation
            T = {}
            T['X'] = G.get_tensor_by_name('X:0')
            T['Q*'] = G.get_tensor_by_name('Q1:0')
            T['Q'] = G.get_tensor_by_name('Q2:0')
            T['a'] = G.get_tensor_by_name('a:0')
            T['J'] = G.get_tensor_by_name('loss:0')
            # Make dictionary to hold sequences of frames
            s = {}
            # Enter training loop, iterate over episodes
            global_step = 0
            epsilon = epsilon0
            cum_rewards = 0
            loss_list = []
            Q_list = []
            reward_list = []
            for ep in range(max_episodes):
                # Reset Pong environment
                obs = env.reset()
                reward = 0
                done = False
                # Add three blank frames and observation to frame_list
                proc_obs = process_raw_frame(obs)
                add_to_frame_list(frame_list, [np.zeros((1,height,width,1)) for i in range(3)]+[proc_obs])
                # Iterate over frames of episode
                frame = 0
                while done == False:
                    frame += 1
                    global_step += 1
                    # Calculate Q from forward pass
                    s['t'] = stack_frames(frame_list[-nframes:])
                    Q1 = sess.run(T['Q*'], feed_dict={T['X']:s['t']})
                    # With probability epsilon select random action, otherwise select argmax(Q)
                    if np.random.rand() < epsilon:
                        action = np.random.choice([1,2,3])
                    else:
                        action = np.argmax(Q1, axis=1).squeeze() + 1
                    # Execute action in environment
                    reward_sum = 0
                    for step in range(n_steps_to_skip):
                        obs, reward, done_, _ = env.step(action)
                        reward_sum += reward
                        if done_ == True:
                            done = True
                    reward = reward_scale*reward_sum
                    cum_rewards += reward
                    if render == True:
                        env.render()
                    # Process observation, add to frame_list, add experience to replay_memory
                    proc_obs = process_raw_frame(obs)
                    add_to_frame_list(frame_list, [proc_obs])
                    s['t+1'] = stack_frames(frame_list[-nframes:])
                    experience = (s['t'], action, reward, s['t+1'], done)
                    add_to_replay_memory(replay_memory, experience)
                    # Sample from replay memory and do backprop
                    mean_batch_loss = 0
                    mean_batch_Q = 0
                    current_Q = np.max(Q1)
                    for experience in sample_from_replay_memory(replay_memory, nsamples=batch_size):
                        # Calculate Q values from updated step
                        s['t'], action, reward, s['t+1'], exp_done = experience
                        Q2 = sess.run(T['Q*'], feed_dict={T['X']:s['t+1']})
                        # Calculate expected Q
                        if exp_done == True:
                            Q_expected = reward
                        else:
                            Q_expected = reward + gamma*np.max(Q2)
                        # Plug into loss and do training op
                        _, loss = sess.run([train_op, T['J']], feed_dict={T['X']:s['t'], T['Q']:Q_expected, T['a']:(action-1)})
                        mean_batch_loss += loss
                        mean_batch_Q += Q_expected
                    # Iterate global variable
                    mean_batch_loss /= batch_size
                    mean_batch_Q /= batch_size
                    print('Episode: {}/{}, frame: {}, batch loss: {:.3e}, mean batch Q: {:.3},\ncurrent max Q: {:.3}, predicted Q: {}'.format(ep+1, max_episodes, frame+1, mean_batch_loss, mean_batch_Q, current_Q, Q1))
                    # Visualize loss, avg Q, blah blah blah
                    # Add random sequence to a 'validation' set that stays fixed once it reaches a max size. Then evaluation of the mean Q on this set will determine how confident the agent is it'll score.
#                    if global_step % 50 == 1:
#                        pass
                    if global_step % plot_every_n_steps == 1:
                        loss_list.append(mean_batch_loss)
                        Q_list.append(mean_batch_Q)
                        reward_list.append(cum_rewards)
                        visualize_training(loss_list, Q_list, reward_list)
                        # Create a dictionary of useful data generated during training. Make it a global variable so it can be accessed in case the training loop exits early.
#                        global final_data = {'num episodes':ep, 'loss':mean_batch_loss, 'Q':mean_batch_Q, 'rewards':reward_list}
                # Anneal epsilon linearly to zero
                epsilon = epsilon0*(1-ep/max_episodes)
                # Save checkpoint if applicable
                if (ep+1) % save_every_n_episodes == 0:
                    print('Saving checkpoint...')
                    saver.save(sess, save_str)
            print('Reached max episodes\nSaving final checkpoint...')
            saver.save(sess, save_str)


''' ======================= TEST RUN AGENT ============================= '''

def model_run():
    ''' Runs the model after training. '''
    save_str = './checkpoints/Pong_2'
    # Initialize Graph, Session, Saver...
    G = initialize_graph()
    with G.as_default():
        saver = tf.train.Saver(var_list=None)



''' ===================== TESTING AND DEBUGGING ======================== '''


#env.reset()
#env.render()
#time.sleep(2)
#action = 0
#while action != 10:
#    try:
#        action = int(input('input action: '))
#    except:
#        env.render(close=True)
#        break
#    obs, reward, done, _ = env.step(action)
#    print('Reward: {}'.format(reward))
##    for i in range(1):
##        obs, reward, done, _ = env.step(1)
#    plt.figure('test')
#    plt.clf()
#    processed_obs = process_raw_frame(obs)
#    plt.imshow(np.squeeze(processed_obs), cmap='gray')
#    plt.draw()
#    env.render()



#g = initialize_graph()
#T = {} # Dictionary of tensors in graph
#T['X'] = g.get_tensor_by_name('X:0')
#T['Q*'] = g.get_tensor_by_name('Q1:0')
#T['loss'] = g.get_tensor_by_name('loss:0')
#T['Q'] = g.get_tensor_by_name('Q2:0')
#
#env.reset()
#obs, reward, done, _ = env.step(1)
#processed_obs = process_raw_frame(obs)
#_, i, j, _ = processed_obs.shape
#frames = np.zeros((1,i,j,4))
#for l in range(4):
#    frames[:,:,:,l] = processed_obs[:,:,:,0]
#
#with g.as_default():
#    with tf.Session(graph=g) as sess:
#        sess.run(tf.global_variables_initializer())
#        print(sess.run(T['Q*'], feed_dict={T['X']:frames}))



model_train(lr=1e-6, max_episodes=100, gamma=np.exp(-1/(4*12)), batch_size=10, epsilon0=0.90, plot_every_n_steps=25, n_steps_to_skip=2, save_every_n_episodes=1, recover_from_last_checkpoint=True, render=False)

#reward = 0
#env.reset()
#while reward == 0:
#    _, reward, _, _ = env.step(0)
#print(reward)

















