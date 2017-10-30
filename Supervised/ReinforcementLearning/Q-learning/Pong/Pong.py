# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:39:27 2017

Implementation of a Q-network which plays Pong (Atari game).

Description:
This project uses a neural network as a function approximator in order to estimate the function Q(s,a) associated with the expected future reward for choosing action a while in state s. Here, s is a sequence of images representing the last 4 frames of a game of Pong, and a is one of {do nothing, up, down}. The input to the network is simply the frame data (representing s), and the output is a 3-D vector Q* representing the expected values of Q for the 3 choices of a. We use an epsilon-Bayesian algorithm to select the action (take random a with prob epsilon, otherwise choose from a distribution softmax(Q)), use that to update the state s -> s', then calculate the expected Q using the Bellman equation Q(s,a) = r + \gamma*argmax(Q*(s',a')), then compute the loss via L = (1/2)*(Q*(s,a) - Q(s,a))^2 (and perform backprop appropriately). This causes the Q-network to learn to approximate Q(s,a), which can then be used to take the optimal action in a given situation.

To do:
    x Change computational graph so that filter size is a larger fraction of the game space (will this be helpful?)
    - Change computational graph so that there are more convolutional layers - want the first layer to detect very simple things like edges/location of ball, second layer will detect stuff like past history of ball/paddles, third will detect relative positioning of ball/paddles
    - Add countdown timer which estimates time left in training
    - Monitor the gradients of the weights during training; let's see
    x Maybe change the way replay memory works so that it only contains examples in which a reward was given (plus surrounding frames). Otherwise, a lot of the frames it doesn't really matter what the agent does because it's just waiting for the opponent to hit the ball back.
    - Why is std(Q*)/max(Q*) so low (~10^-3)? This seems to be independent of the choice of learning rate. How does the agent learn to distinguish between action choices if this is so small? Is there something wrong with the loss function?
    - Why does Q rise to such large values so quickly? Especially values that are much higher than the reward scale.
    - Write a function to save training data between runs (not the weights, TF already takes care of that)
    - Write a function to automatically handle plotting the training data
    - Write functions that:
        1) Collect a series of frames into a static validation set before training start
        2) Run validation data periodically on static validation set and plot <Q>
    - Figure out why the training loss is spiking, and why it spikes to the same value all the time.

Things I (think I) learned:
    1) The discount factor \gamma should be chosen so that \gamma^n = 1/e, where n is the average number of time steps per reward. This makes it so that the characteristic decay timescale of the future rewards is roughly the time between rewards, ensuring the most recent reward is always relevant. (Perhaps I should make \gamma adaptive, which would just remove it as a hyperparameter altogether).
    2) The initialization of the weights in the final layer are very important as well. You want to initialize so that the output is very close to what you would expect to get for losing all your games almost immediately, i.e. if it takes n frames to lose a game quickly, and you choose \gamma^n = 1/e as prescribed above, your expected Q on average is about <Q> ~ \gamma^n r (= -1/e ~ 0.37 in this case). You can do this by making b = <Q> >> rms(W) in the output layer. If not initialized this way, you should see your batch mean <Q> move toward this value, causing large spikes in the training loss along the way. Then as the network improves, Q should ideally increase.

@author: Eric Cotner
"""


import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
import time

''' ===================== SET UP PONG ENVIRONMENT ======================= '''
'''
env = gym.make('Pong-v0')
action_space = env.action_space
# 0,1 = do nothing, 2,4 = up, 3,5 = down (just use 1,2,3)
# These commands actually impart a momentum to the paddle in the specified direction! So you can do 2,2,0,0,0 and actually coast a significant part of the way to the top of the stage!
observation_space = env.observation_space
# Box of shape (210,16,3) = (height,width,color channels)
'''
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
            # Is there some way to vectorize this? (not necessary if ds_factor=1 though)
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
    return (processed_data.reshape(out_shape)-115)/230 # Normalizes pixel values

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

def add_to_list(existing_list, elements_to_add, max_elements_to_keep=1000):
    ''' Adds a set of elements to the list existing_list. Modifies the existing_list IN PLACE, so doesn't return anything. Assumes elements_to_add is a LIST.'''
    assert type(elements_to_add) == list, 'elements_to_add must be a list'
    if len(existing_list) + len(elements_to_add) <= max_elements_to_keep:
        for frame in elements_to_add:
            existing_list.append(frame)
    else:
        for frame in elements_to_add:
            existing_list.append(frame)
        diff = len(existing_list) - max_elements_to_keep
        for i in range(diff):
            existing_list.pop(0)

def add_to_array(array, array_to_add, max_to_keep=1000, axis=-1):
    assert type(array) == np.ndarray, 'array_to_add must be np.ndarray'
    if type(array_to_add) != np.ndarray:
        temp_array = np.append(array, np.array([array_to_add]))
    else:
        temp_array = np.append(array, array_to_add, axis=axis)
    diff = temp_array.shape[axis] - max_to_keep
    if diff > 0:
        temp_array = np.delete(temp_array, np.s_[:diff], axis=axis)
    return temp_array

def stack_frames(frame_list):
    ''' Stacks a list of frames of dimension (1,160,160,1) into a single array of dimension (1,160,160,len(frame_list)). '''
    stack = np.zeros((1,160,160,len(frame_list)))
    for frame_idx in range(len(frame_list)):
        stack[0,:,:,frame_idx] = frame_list[-(1+frame_idx)][0,:,:,0]
    return stack

def add_to_replay_memory(replay_memory, frame, action, reward, done, max_to_keep=4000):
    replay_memory['frames'] = add_to_array(replay_memory['frames'], frame, max_to_keep=max_to_keep, axis=-1)
    replay_memory['actions'] = add_to_array(replay_memory['actions'], action, max_to_keep=max_to_keep, axis=0)
    replay_memory['rewards'] = add_to_array(replay_memory['rewards'], reward, max_to_keep=max_to_keep, axis=0)
    replay_memory['done'] = add_to_array(replay_memory['done'], done, max_to_keep=max_to_keep, axis=0)

def sample_from_replay_memory(replay_memory, nsamples=5, include_most_recent=True, exclude_zero_reward=False):
    ''' Takes nsamples random samples from replay_memory and returns 'samples', a list of tuples, each of which is an experience containing (frames, action, reward, next_frames, done). '''
    s1 = np.zeros((0,160,160,4))
    s2 = np.zeros((0,160,160,4))
    a = np.zeros((0))
    r = np.zeros((0))
    d = np.zeros((0))
    if nsamples >= replay_memory['frames'].shape[-1]-4:
        perm_idx = np.random.permutation(np.arange(4,replay_memory['frames'].shape[-1]))
    elif exclude_zero_reward == False:
        if include_most_recent == True:
            perm_idx = np.random.permutation(np.arange(4,replay_memory['frames'].shape[-1]-1))[:nsamples-1]
            perm_idx = np.append(perm_idx, [replay_memory['frames'].shape[-1]-1])
        else:
            perm_idx = np.random.permutation(np.arange(4,replay_memory['frames'].shape[-1]))[:nsamples]
    elif exclude_zero_reward == True:
        nonzero_rewards_idx = np.nonzero(replay_memory['rewards'][4:])[0]+4
        if nsamples >= nonzero_rewards_idx.shape[0]:
            perm_idx = np.random.permutation(nonzero_rewards_idx)
        elif include_most_recent == True:
            perm_idx = np.random.permutation(nonzero_rewards_idx[:-1])[:nsamples-1]
            perm_idx = np.append(perm_idx, nonzero_rewards_idx[-1])
        else:
            perm_idx = np.random.permutation(nonzero_rewards_idx[:-1])[:nsamples]
    for idx in perm_idx:
        s1 = np.append(s1, replay_memory['frames'][:,:,:,(idx-4):idx], axis=0)
        s2 = np.append(s2, replay_memory['frames'][:,:,:,(idx-4)+1:idx+1], axis=0)
        a = np.append(a, [replay_memory['actions'][idx]], axis=0)
        r = np.append(r, [replay_memory['rewards'][idx]], axis=0)
        d = np.append(d, [replay_memory['done'][idx]], axis=0)
    return (s1, s2, a, r, d)



''' =================== SET UP COMPUTATIONAL GRAPH ===================== '''

def initialize_graph(in_height=160, in_width=160, color=1, nframes=4):
    '''
    Initializes the computational graph for the neural network. The arguments are pretty self-explanatory, but should not be modified since I'm not sure how the padding on the convolutional layers works exactly, so the number of nodes in the dense layer after the 2nd convolutional layer is hard-coded based on the default arguments; this number will need to be changed if the input dimensions are changed. The output is 'G', a tensorflow Graph object containing the computational graph. All the tensor declarations are local to this function which avoids polluting the global namespace, but can still be accessed using the Graph.get_tensor_by_name() method. (Need to make sure all tensors of interest have names!)
    '''
    
    def lrelu(tensor_in, name=None):
        return tf.maximum(0.1*tensor_in, tensor_in, name=name)
    
    tf.reset_default_graph()
    G = tf.Graph()
    
    with G.as_default():
        xavier_init = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        
        # Input layer
        X = tf.placeholder(dtype=tf.float32, shape=[None,in_height,in_width,nframes*color], name='X')
        
        # Convolutional and pooling layer(s)
        W1 = tf.get_variable(shape=[32,32,nframes*color,32], initializer=xavier_init, name='W1')
        b1 = tf.Variable(0.01*np.ones((32,)), dtype=tf.float32, name='b1')
        conv1 = lrelu(tf.nn.conv2d(X, W1, strides=[1,2,2,1], padding='SAME')+b1, name='conv1')
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
        
        W2 = tf.get_variable(shape=[11,11,32,32], initializer=xavier_init, name='W2')
        b2 = tf.Variable(0.01*np.ones((32,)), dtype=tf.float32, name='b2')
        conv2 = lrelu(tf.nn.conv2d(pool1, W2, strides=[1,2,2,1], padding='SAME')+b2, name='conv2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
        
        # Dense layer(s)
        pool2_flat = tf.reshape(pool2, shape=[-1,3200], name='pool2_flat')
        W3 = tf.get_variable(shape=[3200,256], initializer=xavier_init, name='W3')
        b3 = tf.Variable(0.01*np.ones((256,)), dtype=tf.float32, name='b3')
        A3 = tf.nn.relu(tf.matmul(pool2_flat,W3)+b3, name='A3')
        
        # Output layer
        # Want to initialize these variables so that they're close to the expected Q for constantly losing games, so a bias of around b = -1/e, and weights such that b >> |W|
        W4 = tf.Variable(0.01*np.random.randn(256,3), dtype=tf.float32, name='W4')
        b4 = tf.Variable(-np.exp(-1)*np.ones((3,)), dtype=tf.float32, name='b4')
        Q1 = tf.add(tf.matmul(A3,W4),b4, name='Q1')
        
        '''
        # Single hidden layer, comment out above stuff to use
        W1 = tf.Variable(0.01*np.random.randn(160,160,4,25), dtype=tf.float32, name='W')
        b1 = tf.Variable(np.zeros((25)), dtype=tf.float32, name='b')
        A1 = lrelu(tf.einsum('ijkl,jklm->im',X,W1)+b1, name='A1')
        W2 = tf.Variable(0.01*np.random.randn(25,3), dtype=tf.float32, name='W2')
        b2 = tf.Variable(np.zeros((3)), dtype=tf.float32, name='b2')
        Q1 = tf.add(tf.matmul(A1,W2), b2, name='Q1')
        '''
        
        # Define predictions, loss, and accuracy metrics
        Q2 = tf.placeholder(tf.float32, shape=[None], name='Q2')
        a = tf.placeholder(tf.int32, shape=[None], name='a')
        
        # Need to make mask tensor so that selcted elements of Q1 and Q2 are same size
        Q1_mask = tf.one_hot(a, depth=3, dtype=tf.float32, axis=-1)
        loss = tf.multiply(0.5,tf.reduce_mean((tf.reduce_sum(Q1_mask*Q1, axis=1) - Q2)**2), name='loss')
         
    # Output Graph object
    return G

''' ================== SET UP OPTIMIZER AND TRAINING OPERATIONS =========== '''

def initialize_train_op(graph, lr):
    with graph.as_default():
#        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
#        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(graph.get_tensor_by_name('loss:0'))
    return train_op

''' ===================== TRAINING FUNCTIONS ============================= '''

def visualize_training(loss_list=[], Q_list=[], reward_list=[], avg_reward_list=[]):
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
    if (reward_list != []) or (avg_reward_list != []):
        plt.figure('Rewards')
        plt.clf()
        if reward_list != []:
            plt.plot(reward_list, label='Rewards')
        if avg_reward_list != []:
            plt.plot(avg_reward_list, label='Moving average')
        plt.title('Rewards over time')
        plt.xlabel('Time (arb. units)')
        plt.ylabel('Rewards')
        plt.legend()
        plt.draw()
        plt.pause(1e-10)

def softmax(v, axis=0):
    vmax = np.max(v)
    exp = np.exp(v-vmax)
    return exp/np.sum(exp, axis=axis)

def model_train(lr, max_episodes, gamma, batch_size=5, epsilon0=0.9, plot_every_n_steps=100, n_steps_to_skip=2, save_every_n_episodes=5, recover_from_last_checkpoint=False, render=False):
    '''
    Training loop for the Q-network. Automatically initializes computational graph architecture, refreshes checkpoints if applicable, and runs the training loop over the required number of episodes. The training loop basically iterates through an episode frame by frame, at each step predicting the Q value for each action, given the previous 4 frames of the sequence. The action with the highest Q is picked (there is an epsilon chance of this decision being random), and then this action is used to step to the next frame. The Q values of the next frame are used to compute the expected Q, and then the Q-network weights are updated by backprop using a MSE loss.
    '''
    # Define some constants and set up random stuff
    nframes = 4
    height, width = (160,160)
    save_str = './checkpoints/Pong_7'
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
            # Set up initial frame_list and replay_memory
            replay_memory = {'frames':np.zeros((1,height,width,0)), 'actions':np.zeros((0)), 'rewards':np.zeros((0)), 'done':np.zeros((0), dtype=bool)}
            # Make dictionary of tensors in graph needed for computation
            T = {}
            T['X'] = G.get_tensor_by_name('X:0')
            T['Q*'] = G.get_tensor_by_name('Q1:0')
            T['Q'] = G.get_tensor_by_name('Q2:0')
            T['a'] = G.get_tensor_by_name('a:0')
            T['J'] = G.get_tensor_by_name('loss:0')
            # Make dictionary to hold temporary sequences of frames
            s = {}
            # Enter training loop, iterate over episodes
            global_step = 0
            epsilon = epsilon0
            cum_rewards = 0
            avg_reward = -1
            avg_param = np.exp(-1/20) # Sets decay time of reward moving average
            loss_list = []
            Q_list = []
            reward_plot_list = []
            avg_reward_list = []
            quit_flag = False
            for ep in range(max_episodes):
                # Reset Pong environment
                obs = env.reset()
                # Add three blank frames and processed observation to replay memory
                action = np.ones((4))
                reward = np.zeros((4))
                done = np.zeros((4), dtype=bool)
                proc_obs = process_raw_frame(obs)
                add_to_replay_memory(replay_memory, np.append(np.zeros((1,height,width,3)), proc_obs, axis=-1), action, reward, done)
                done = False
                # Iterate over frames of episode
                frame = 0
                while done == False:
                    frame += 1
                    global_step += 1
                    # Calculate Q from forward pass
                    s['t'] = replay_memory['frames'][:,:,:,-nframes:]
                    Q1 = sess.run(T['Q*'], feed_dict={T['X']:s['t']})
                    if np.any(np.isnan(Q1)) == True:
                        print('Detected nan in Q*, exiting training')
                        quit_flag = True
                        break
                    # With probability epsilon select random action, otherwise select action from softmax distribution over Q*
                    if np.random.rand() < epsilon:
                        action = np.random.choice([1,2,3])
                    else:
                        softmaxQ = softmax(Q1, axis=1).squeeze()
                        action = np.random.choice([1,2,3], p=softmaxQ)
                    # Execute action in environment
                    reward_sum = 0
                    for step in range(n_steps_to_skip+1):
                        if step == 0:
                            obs, reward, done_, _ = env.step(action)
                        else:
                            obs, reward, done_, _ = env.step(1) # Do nothing in between skips, otherwise it overshoots
                        reward_sum += reward
                        if done_ == True:
                            done = True
                    reward = reward_scale*reward_sum
                    cum_rewards += reward
                    if render == True:
                        env.render()
                    # Process observation, add to frame_list, add experience to replay_memory
                    proc_obs = process_raw_frame(obs)
                    add_to_replay_memory(replay_memory, proc_obs, action, reward, done)
                    # Sample from replay memory and do backprop
                    current_action = action
                    
                    # Train over minibatches from replay memory
                    s['t'], s['t+1'], actions, rewards, exp_done = sample_from_replay_memory(replay_memory, nsamples=batch_size)
                    Q2 = sess.run(T['Q*'], feed_dict={T['X']:s['t+1']})
                    # Calculate expected Q
                    Q_expected = rewards + (1-exp_done)*gamma*np.max(Q2, axis=1)
                    # Plug into loss and do training
                    _, loss = sess.run([train_op, T['J']], feed_dict={T['X']:s['t'], T['Q']:Q_expected, T['a']:(actions-1)})
                    mean_batch_loss = np.mean(loss)
                    mean_batch_Q = np.mean(Q_expected)
                    print('Episode: {}/{},\tframe: {},\tbatch loss: {:.3e},\tmean batch Q: {:.3e},\ncurrent max(Q*): {:.3e},\tcurrent action: {},\tcurrent std(Q*)/max(Q*): {:.3e}'.format(ep+1, max_episodes, frame+1, mean_batch_loss, mean_batch_Q, np.max(Q1), current_action, np.std(Q1)/np.max(Q1)))
                    # Visualize loss, avg Q, blah blah blah
                    if global_step % plot_every_n_steps == 1:
                        loss_list.append(mean_batch_loss)
                        Q_list.append(mean_batch_Q)
                        reward_plot_list.append(cum_rewards)
                        avg_reward = (1-avg_param)*cum_rewards + avg_param*avg_reward
                        cum_rewards = 0
                        avg_reward_list.append(avg_reward)
                        visualize_training(loss_list, Q_list, reward_plot_list, avg_reward_list)
                if quit_flag == True:
                    break
                # Anneal epsilon linearly to zero
                epsilon = epsilon0*(1-ep/max_episodes)
                # Save checkpoint if applicable
                if (ep+1) % save_every_n_episodes == 0:
                    print('Saving checkpoint...')
                    saver.save(sess, save_str)
            if quit_flag == True:
                pass
            else:
                print('Reached max episodes\nSaving final checkpoint...')
                saver.save(sess, save_str)


''' ======================= RUN AGENT ============================= '''

def model_run(n_steps_to_skip = 0, epsilon=0):
    ''' Runs the model after training. '''
    save_str = './checkpoints/Pong_7'
    nframes = 4
    # Initialize Graph, Session, Saver...
    G = initialize_graph()
    with G.as_default():
        # Create dictionary of tensors in graph
        T = {}
        T['X'] = G.get_tensor_by_name('X:0')
        T['Q*'] = G.get_tensor_by_name('Q1:0')
        T['loss'] = G.get_tensor_by_name('loss:0')
        T['Q'] = G.get_tensor_by_name('Q2:0')
        saver = tf.train.Saver(var_list=None)
        with tf.Session() as sess:
            # Recover from last checkpoint
            print('Recovering from last checkpoint...')
            saver.restore(sess, save_str)
            # Enter the play loop
            done_playing = False
            while done_playing == False:
                obs = env.reset()
                proc_obs = process_raw_frame(obs)
                frames = np.zeros((1,160,160,3))
                frames = add_to_array(frames, proc_obs, max_to_keep=4, axis=-1)
                done = False
                # Iterate over frames of episode
                frame = 0
                while done == False:
                    # Feed the sequence into the network, get Q*
                    Q = sess.run(T['Q*'], feed_dict={T['X']:frames})
                    print('Q: {}, std(Q)/mean(Q): {:.3e}'.format(Q, np.std(Q)/np.mean(Q)))
                    # Select argmax(Q*) as action
                    if np.random.rand() < epsilon:
                        action = np.random.choice([1,2,3])
                    else:
#                        softmaxQ = softmax(Q.squeeze())
#                        action = np.random.choice([1,2,3], p=softmaxQ)
                        action = np.argmax(Q.squeeze())+1
                    print('action: {}'.format(action))
                    # Step to the next state with the selected action, observe next state, render image
                    reward_sum = 0
                    for step in range(n_steps_to_skip+1):
                        if step == 0:
                            obs, reward, done_, _ = env.step(action)
                        else:
                            obs, reward, done_, _ = env.step(1) # Do nothing in between skips, otherwise it overshoots
                        reward_sum += reward
                        if done_ == True:
                            done = True
                    env.render()
                    # Add state to frame_list and build next sequence
                    proc_obs = process_raw_frame(obs)
                    frames = add_to_array(frames, proc_obs, max_to_keep=4, axis=-1)
#                    print(frames.shape)
                # Ask whether to play another game
                done_playing = False
                valid_input = False
                while valid_input == False:
                    play_again = input('Play again? [y/n]: ')
                    if play_again.lower() == 'y':
                        done_playing = False
                        valid_input = True
                    elif play_again.lower() == 'n':
                        done_playing = True
                        valid_input = True
                        env.render(close=True)
                    else:
                        valid_input = False
                


''' ===================== TESTING AND DEBUGGING ======================== '''

def visualize_conv_filters():
    save_str = './checkpoints/Pong_7'
    # Build the computational graph
    G = initialize_graph()
    with G.as_default():
        # Import the weights
        W1 = G.get_tensor_by_name('W1:0')
        saver = tf.train.Saver(var_list=[W1])
        with tf.Session() as sess:
            saver.restore(sess, save_str)
            # Get the weights of the first convolutional filter layer
            F = sess.run(W1)
            # Plot them all side by side (64 = 8*8)
            plt.figure('Filters')
            for i in range(8):
                for j in range(4):
                    f = F[:,:,:,i+8*j]
                    plt.subplot(8, 4, 1+i+8*j)
                    plt.imshow(f)
            plt.draw()

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



#model_train(lr=3e-3, max_episodes=100, gamma=np.exp(-1/(5*12)), batch_size=10, epsilon0=0, plot_every_n_steps=25, n_steps_to_skip=1, save_every_n_episodes=1, recover_from_last_checkpoint=True, render=False)


#model_run(n_steps_to_skip=1, epsilon=0.1)

#visualize_conv_filters()


#reward = 0
#env.reset()
#while reward == 0:
#    _, reward, _, _ = env.step(0)
#print(reward)

















