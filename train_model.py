import multiprocessing
import threading
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
from time import sleep
import random
from itertools import product

from animalai.envs.arena_config import ArenaConfig
from animalai.envs import UnityEnvironment

OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
#N_WORKERS = 1
MAX_GLOBAL_EP = 10
GLOBAL_NET_SCOPE = 'Global_Net'
MAX_STEPS = 50000
UPDATE_GLOBAL_ITER = 1000
GAMMA = 0.8
#GAMMA = 0.5
ENTROPY_BETA = 0.01
#ENTROPY_BETA = 0.001
LR_A = 0.01    # lr actor
LR_C = 0.01    # lr critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

ACTIONS = [list(elem) for elem in product([0, 1, 2], repeat=2)][:-3]
ARENA = ArenaConfig('configs/3-Obstacles.yaml')

N_S = 21168
N_A = len(ACTIONS)

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info = None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std/np.sqrt(np.square(out).sum(axis=0,keepdims=True))
        return tf.constant(out)
    return _initializer
            
class ACNet(object):
    def __init__(self, scope, globalAC=None):
        
        if scope == GLOBAL_NET_SCOPE:   # глобальная сеть
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # функция потерь
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                self.actions_onehot = tf.one_hot(self.a_his, N_A, dtype=tf.float32)

                self.state_init, self.state_in, self.state_out, self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)
                
                advantages = tf.subtract(self.v_target, self.v, name='TD_error')

                with tf.name_scope('c_loss'):
                    self.c_loss = 0.5 * tf.reduce_sum(tf.square(advantages))

                with tf.name_scope('a_loss'):
                    responsible_outputs = tf.reduce_sum(self.a_prob * self.actions_onehot,[1])
                    
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob))
                    exp_v = tf.log(responsible_outputs) * advantages
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
                    
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
                    
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
                    
    def _build_net(self, scope):
        # обработка картинки
        x_shaped = tf.reshape(self.s, [-1, 84, 84, 3])

        # инициализация весов сверточной сети
        w1 = tf.Variable(tf.random_normal([8,8,3,32]))
        w2 = tf.Variable(tf.random_normal([8,8,32,64]))
        b1 = tf.Variable(tf.random_normal([32]))
        b2 = tf.Variable(tf.random_normal([64]))

        # сверточная сеть
        conv_1 = tf.nn.relu(tf.nn.conv2d(x_shaped, w1, strides=[1, 1, 1, 1], padding='VALID') + b1)
        maxpool_1 = tf.nn.max_pool(conv_1, ksize=[1, 8, 8, 1], strides=[1, 4, 4, 1], padding='VALID')
        conv_2 = tf.nn.relu(tf.nn.conv2d(maxpool_1, w2, strides=[1, 1, 1, 1], padding='VALID') + b2)
        maxpool_2 = tf.nn.max_pool(conv_2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID')

        # полносвязный слой
        hidden = slim.fully_connected(slim.flatten(maxpool_2), 1024, activation_fn=tf.nn.elu)

        # инициализация весов рекуррентной сети
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(1024, state_is_tuple=True)
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        state_init = [c_init, h_init]

        # рекуррентная сеть
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        state_in = (c_in, h_in)
        rnn_in = tf.expand_dims(hidden, [0])
        step_size = tf.shape(x_shaped)[:1]
        state_in1 = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm_cell, rnn_in, initial_state=state_in1, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        state_out=(lstm_c[:1, :], lstm_h[:1,:])
        rnn_out=tf.reshape(lstm_outputs, [-1, 1024])
    
        # ценность политики и ценность состояния
        with tf.variable_scope('actor'):
            a_prob = slim.fully_connected(rnn_out, N_A,
                                          activation_fn=tf.nn.softmax,
                                          weights_initializer=normalized_columns_initializer(0.01),
                                          biases_initializer=None)

        with tf.variable_scope('critic'):
            v = slim.fully_connected(rnn_out, 1, activation_fn=None,
                                     weights_initializer=normalized_columns_initializer(1.0),
                                     biases_initializer=None)

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return state_init, state_in, state_out, a_prob, v, a_params, c_params

    def update_global(self, feed_dict): 
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # применение локальных градиентов к глобальной сети

    def pull_global(self): 
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, rnn_state):
        prob_weights, state_out = SESS.run([self.a_prob, self.state_out], feed_dict={self.s: s,
                                                                                     self.state_in[0]:rnn_state[0],
                                                                                     self.state_in[1]:rnn_state[1]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return [action, ACTIONS[action], state_out, prob_weights]
    
class Worker(object):
    def __init__(self, name, globalAC):
        env_id = int(name[-1])
        self.env = UnityEnvironment(file_name='env/AnimalAI', 
                                    worker_id=env_id, seed=0, 
                                    docker_training=False, 
                                    n_arenas=1, play=False, 
                                    inference=True, 
                                    resolution=None)

        reset = self.env.reset(train_mode=True)
        
        self.name = name
        self.AC = ACNet(name, globalAC)
        
    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            #reset = self.env.reset(train_mode=True)
            reset = self.env.reset(train_mode=True, arenas_configurations=ARENA)
            
            brain = reset['Learner']
            s = np.array(brain.visual_observations, dtype='float32').reshape(84, 84, 3).flatten()[np.newaxis, :]
            ep_r = 0
            
            rnn_state=self.AC.state_init
            
            for ep_t in range(MAX_STEPS):
                a = self.AC.choose_action(s, rnn_state)
                rnn_state = a[2]

                
                if a[0] == 0:
                    info = [self.env.step(vector_action=[0,1]) for i in range(30)][-1]
                else: 
                    info = self.env.step(vector_action=a[1])
                brain = info['Learner']
                s_ = np.array(brain.visual_observations, dtype='float32').reshape(84, 84, 3).flatten()[np.newaxis, :]
                r = brain.rewards[0]
                done = brain.local_done[0]
                
                end = True if (ep_t == MAX_STEPS - 1) else False
                if r == 0: r = -0.0125    
                ep_r += r

                buffer_s.append(s)
                buffer_a.append(a[0])
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or end:   # обновление сети
                    if end:
                        v_s_ = 0  
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_, 
                                                    self.AC.state_in[0]:rnn_state[0],
                                                    self.AC.state_in[1]:rnn_state[1]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        self.AC.state_in[0]:rnn_state[0],
                        self.AC.state_in[1]:rnn_state[1]
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                
                if end:
                    if len(GLOBAL_RUNNING_R) == 0:  # запись наград эпизода
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    break
                    
if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # id
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
