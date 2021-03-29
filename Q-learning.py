import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from SAE import SAE, Train
from pre_processing import preprocessing
import random
import matplotlib.pyplot as plt

class DQN(object):
    def __init__(self, X_train, y_train, X_val, y_val, path=None, MEMORY_CAPACITY=64, taget_replace_iter = 5, gamma = 0, epsilon = 0.9, lr = 0.001):
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.FloatTensor(y_val)
        self.class_num = len(np.unique(self.y_train))

        self.eval_net = SAE(input_dim=self.X_train.shape[1], output_dim=self.class_num, pre=False, path=path)
        self.target_net = SAE(input_dim=self.X_train.shape[1], output_dim=self.class_num, pre=False, path=path)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter = 0  # 如果次数到了，更新target_net
        self.memory_counter = 0  # for storing memory
        self.state_dim = self.X_train.shape[1]
        self.target_replace_iter, self.gamma, self.epsilon = taget_replace_iter, gamma, epsilon
        self.memory_capacity = MEMORY_CAPACITY
        self.memory = np.zeros((MEMORY_CAPACITY, self.state_dim * 2 + 2))  # 初始化记忆
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.epsilon: 
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]# return the argmax index
        else:  # random
            action = np.random.randint(0, self.class_num)
        return action

  
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_)) 
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        b_memory = self.memory[:, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_dim])
        b_a = torch.LongTensor(b_memory[:, self.state_dim:self.state_dim + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.state_dim + 1:self.state_dim + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.state_dim:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.memory_capacity, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class env:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.size = self.X_train.shape[0]

    def reset(self):
        self.shuffle()
        idx = np.random.randint(0, self.size)
        return self.X_train[idx], self.y_train[idx]

    def shuffle(self):
        index = np.arange(len(self.X_train))
        random.shuffle(index)
        self.X_train, self.y_train = self.X_train[index], self.y_train[index]

    def step(self, action, label):
        idx = np.random.randint(0, self.size)
        if action == label:
            reward = 1
        else:
            reward = -1
        return self.X_train[idx], reward, self.y_train[idx]
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def split(ratio, X_train, y_train):
    index = np.arange(0, len(X_train))
    random.shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]

    num_of_train = int(ratio*len(X_train))
    return X_train[0:num_of_train], y_train[0:num_of_train], X_train[num_of_train:], y_train[num_of_train:]

if __name__ == '__main__':
    setup_seed(1000)
    dataset = preprocessing('DataSetA.mat')
    X_train, X_test, y_train, y_test = dataset.Data_Preprocess(slice_size=2400, L=2400, window=False)
    train = Train(X_train, y_train, X_test, y_test, pre=True)
    train.train()
    path = train.get_para()
    ######copy parameters#######

    dqn = DQN(X_train, y_train, X_test, y_test, path=path)
    X_train, y_train, X_val, y_val = split(0.8, X_train, y_train)
    env = env(X_train, y_train, X_val, y_val)
    # assert 5==6
    MEMORY = 64
    epi = []
    for i_episode in range(200):
        s, label = env.reset()  # 搜集当前环境状态。
        len_episode = 0
        ep_r = 0
        correct = 0
        while True:
            a = dqn.choose_action(s)

            # take action
            s_, r, label_ = env.step(a, label)
            if len_episode == 512:
                done = True
            else:
                done = False
            if r == 1:
                correct+=1
            dqn.store_transition(s, a, r, s_)

            ep_r += r
            if dqn.memory_counter > MEMORY:
                dqn.learn()
                if done:
                    epi.append(round(ep_r, 2))
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2),'| acc: ', round(correct/512, 2))

            if done:
                break
            len_episode+=1
            s = s_
            label = label_
    plt.plot(epi)
    plt.show()
