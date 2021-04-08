import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from SAE import SAE, Train
from pre_processing import preprocessing
import random
import optuna
import torch.optim as optim
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import StratifiedKFold
from Crossprocessing import Crosspreprocessing
# from cross-validation-preprocessing import
# 定义参数
# BATCH_SIZE = 32  # 每一批的训练量
# LR = 0.001  # 学习率
# EPSILON = 0.9  # 贪婪策略指数，Q-learning的一个指数，用于指示是探索还是利用。
# GAMMA = 0  # reward discount
# TARGET_REPLACE_ITER = 5  # target的更新频率

class DQN(object):
    def __init__(self, env, path=None, MEMORY_CAPACITY=64, taget_replace_iter=5, gamma=0.1,
                 epsilon=0.95, lr=0.001):
        self.env = env
        self.class_num = self.env.class_num
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.eval_net = SAE(input_dim=self.env.dimension, output_dim=self.class_num, pre=False, path=path).to(self.device)
        self.target_net = SAE(input_dim=self.env.dimension, output_dim=self.class_num, pre=False, path=path).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter = 0  # 如果次数到了，更新target_net
        self.memory_counter = 0  # for storing memory
        self.state_dim = self.env.dimension
        self.target_replace_iter, self.gamma, self.epsilon = taget_replace_iter, gamma, epsilon
        self.memory_capacity = MEMORY_CAPACITY
        self.memory = np.zeros((MEMORY_CAPACITY, self.state_dim * 2 + 2))  # 初始化记忆
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        print("------------------current parameter [ gamma:", gamma, "epsilon:", epsilon, "lr:", lr,
              "]-----------------------")

    # 选择动作
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.device)
        # input only one sample
        if np.random.uniform() < self.epsilon:  # 贪婪策略
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]  # return the argmax index
        else:  # random
            action = np.random.randint(0, self.class_num)
        return action

    # 存储记忆
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # 将每个参数打包起来
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 学习过程
        # sample_index = np.random.choice(self.memory_capacity, self.memory_capacity)
        b_memory = self.memory[:, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_dim]).to(self.device)
        b_a = torch.LongTensor(b_memory[:, self.state_dim:self.state_dim + 1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(b_memory[:, self.state_dim + 1:self.state_dim + 2]).to(self.device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.state_dim:]).to(self.device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach的作用就是不反向传播去更新，因为target的更新在前面定义好了的
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.memory_capacity, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def validation(self):
        with torch.no_grad():
            X, y = self.env.get_valdata()
            X = torch.FloatTensor(X).to(self.device)
            y = torch.FloatTensor(y).to(self.device)
            pred = self.eval_net(X)
            pred = torch.argmax(pred, dim=1)
            current_num = torch.eq(pred, y).sum().item()
            val_acc = current_num/len(X)

            return round(val_acc, 4)


class env:
    def __init__(self, X_train, y_train, kf):
        self.X_train = X_train
        self.y_train = y_train
        # self.size = self.X_train.shape[0]
        self.dimension = self.X_train.shape[1]
        self.class_num = len(np.unique(self.y_train))
        self.kf = kf
        self.X, self.Y = None, None
        self.X_val, self.Y_val = None, None
        self.ff = 1

    def cross_val(self):
        try:
            train_idx, val_idx = next(self.kf)
        except:
            print("----------")
        self.X, self.Y = self.X_train[train_idx], self.y_train[train_idx]
        self.X_val, self.Y_val = self.X_train[val_idx], self.y_train[val_idx]


    def reset(self):
        self.shuffle()
        idx = np.random.randint(0, self.X.shape[0])
        return self.X[idx], self.Y[idx]

    def shuffle(self):
        index = np.arange(len(self.X))
        random.shuffle(index)
        self.X, self.Y = self.X[index], self.Y[index]

    def step(self, action, label):
        idx = np.random.randint(0, self.X.shape[0])
        if action == label:
            reward = 1
        else:
            reward = -1
        return self.X[idx], reward, self.Y[idx]

    def get_valdata(self):
        return self.X_val, self.Y_val





class BayesianSearchV2:
    def __init__(self, path_para,path_encoder, X_train, y_train, memory):
        self.model = None
        self.env = None
        self.X_train, self.y_train = X_train, y_train
        self.memory = memory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.x_train, self.y_train, self.x_val, self.y_val = x_train, y_train, x_val, y_val
        self.path_para = path_para
        self.path_encoder = path_encoder
        self.parameterization = None
        self.best_model = None
        # self.kf = kf

    def define_model(self, trial):
        gamma_selection = self.parameterization.get("gamma", [0])
        gamma = trial.suggest_uniform("gamma", gamma_selection[0], gamma_selection[-1])
        lr_selection = self.parameterization.get("lr", [0.0001])
        lr = trial.suggest_loguniform("lr", lr_selection[0], lr_selection[-1])
        epsilon_selection = self.parameterization.get("epsilon", [0.9])
        epsilon = trial.suggest_loguniform("epsilon", epsilon_selection[0], epsilon_selection[-1])
        self.model = DQN(self.env, path=self.path_encoder, gamma=gamma, lr=lr, epsilon=epsilon)


    #     return self.model

        # hidden = self.parameterization.get("num_hidden", [1])
        # num_hidden = trial.suggest_int("num_hidden", hidden[0], hidden[-1])
        # layers = []
        # func = self.parameterization.get("activate_func", ["ReLU"])
        # activate_func = trial.suggest_categorical("activate_func", func)
        # last_dim = self.in_features
        # hidden_range = self.parameterization.get("hidden_dim_range", [500])
        # for i in range(num_hidden):
        #     hidden_dim = trial.suggest_int("n_units_l{}".format(i), hidden_range[0], hidden_range[-1])
        #     layers.append(nn.Linear(last_dim, hidden_dim))
        #     layers.append(getattr(nn, activate_func)())
        #     last_dim = hidden_dim
        # layers.append(nn.Linear(last_dim, self.out_features))
    #
    #
    # def shuffle_data(self):
    #     indices = np.arange(self.x_train.shape[0])
    #     random.shuffle(indices)
    #     return self.x_train[indices], self.y_train[indices]

    def callback(self, study, trial):
        if study.best_trial == trial:
            self.best_model = self.model

    def objective(self, trial):
        # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

        # self.model = self.model.to(self.device)
        kf = StratifiedKFold(n_splits=10).split(self.X_train, self.y_train)
        self.env = env(self.X_train, self.y_train, kf)
        epochs = self.parameterization.get("epochs", [200])
        num_epochs = trial.suggest_int("epochs", epochs[0], epochs[-1])
        train_acc = np.zeros(num_epochs)
        epi = np.zeros(num_epochs)
        val_acc = np.zeros(num_epochs)
        for cross_val in range(10):
            self.define_model(trial)
            self.env.cross_val()
            # pr("cross", cross_val)
            for i_episode in range(num_epochs):
                s, label = self.env.reset()
                len_episode = 0
                ep_r = 0
                correct = 0
                while True:
                    a = self.model.choose_action(s)

                    # take action
                    s_, r, label_ = self.env.step(a, label)
                    if len_episode == 511:
                        done = True
                    else:
                        done = False
                    if r == 1:
                        correct += 1
                    self.model.store_transition(s, a, r, s_)

                    ep_r += r
                    if self.model.memory_counter > self.memory:
                        self.model.learn()
                        val_ac = self.model.validation()
                        if done:
                            val_acc[i_episode] += val_ac
                            epi[i_episode] += ep_r
                            train_acc[i_episode] += correct/512
                            print('Cross Validation:', cross_val,' Ep: ', i_episode,
                                  '| Ep_r: ', round(ep_r, 2), '| acc: ', round(correct / 512, 2),'| val_acc: ', val_ac)

                    if done:
                        break
                    len_episode += 1
                    s = s_
                    label = label_
        # print("--------finish training fault detection system----------------")

        val_acc = val_acc/10
        epi = epi/10
        train_acc = train_acc/10

        return np.mean(val_acc[-10:])

        # for _ in range(num_epochs):
        #     x, y = self.shuffle_data()
        #     for i in range(self.x_train.shape[0] // batch_size):
        #         optimizer.zero_grad()
        #         outputs = self.model(x[i * batch_size: (i + 1) * batch_size])
        #         loss = criterion(outputs, y[i * batch_size: (i + 1) * batch_size])
        #         loss.backward()
        #         optimizer.step()
        #
        # with torch.no_grad():
        #     Y = self.model(self.x_val)
        #     loss_ = criterion(Y, self.y_val)
        #     # when search for optimal model, some parameters may cause
        #     if np.isnan(loss_.item()):
        #         return 10000.0
        #     else:
        #         return loss_.item()

    def load_json(self):
        if self.path_para.endswith('.json'):
            with open(self.path_para, 'r', encoding='utf8') as load_f:
                load_dict = json.load(load_f)
                self.parameterization = load_dict

        else:
            print('Please use json file.')

    def optimize_para(self, visualize=True):
        self.load_json()
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=10, callbacks=[self.callback])

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        print(trial.params)
        # for key, value in trial.params.items():
        #     print("    {}: {}".format(key, value))
        # if visualize:
        #     with torch.no_grad():
        #         y_pred = self.best_model(self.x_val)
        #         y_pred = y_pred.cpu().detach().numpy()
        #         x_data = self.x_val.cpu().detach().numpy()
        #         y_data = self.y_val.cpu().detach().numpy()
        #         plt.cla()
        #         plt.scatter(x_data, y_data)
        #         plt.plot(x_data, y_pred, color='r', lw=3)
        #         plt.show()





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

    num_of_train = int(ratio * len(X_train))
    return X_train[0:num_of_train], y_train[0:num_of_train], X_train[num_of_train:], y_train[num_of_train:]

def shuffle(X, Y):
    index = np.arange(len(X))
    random.shuffle(index)
    # self.X, self.Y = self.X[index], self.Y[index]
    return X[index], Y[index]

if __name__ == '__main__':
    setup_seed(1000)
    dataset = Crosspreprocessing('DataSetA.mat')
    X_train, y_train = dataset.Data_Preprocess(slice_size=2400, L=2400, window=False)
    train = Train(X_train, y_train, pre=True)
    train.train()
    path = train.get_para()
    ######copy parameters#######
    # X_train, y_train, X_val, y_val = split(0.75, X_train, y_train)
    # kf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True).split(X_train, y_train)
    # env = env(X_train, y_train, kf)
    X_train, y_train = shuffle(X_train, y_train)
    search = BayesianSearchV2("para.json", path, X_train, y_train, 64)
    search.optimize_para()

    # epi = []
    # acc_ = []
    # print("--------start training fault detection system----------------")
    # for i_episode in range(200):
    #     s, label = env.reset()
    #     len_episode = 0
    #     ep_r = 0
    #     correct = 0
    #     while True:
    #         a = dqn.choose_action(s)
    #
    #         # take action
    #         s_, r, label_ = env.step(a, label)
    #         if len_episode == 512:
    #             done = True
    #         else:
    #             done = False
    #         if r == 1:
    #             correct += 1
    #         dqn.store_transition(s, a, r, s_)
    #
    #         ep_r += r
    #         if dqn.memory_counter > MEMORY:
    #             dqn.learn()
    #             if done:
    #                 epi.append(round(ep_r, 2))
    #                 acc_.append(round(correct / 512, 2))
    #                 print('Ep: ', i_episode,
    #                       '| Ep_r: ', round(ep_r, 2), '| acc: ', round(correct / 512, 2))
    #         if done:
    #             break
    #         len_episode += 1
    #         s = s_
    #         label = label_
    # print("--------finish training fault detection system----------------")
    # plt.figure()
    # plt.plot(epi)
    # plt.show()
    # plt.figure()
    # plt.plot(acc_)
    # plt.show()

    # s =   # 搜集当前环境状态。
    # ep_r = 0
    # episode_length = 0
    # while True:
    #     env.render()
    #     a = dqn.choose_action(s)
    #
    #     # take action
    #     s_, r, done, info = env.step(a)
    #
    #     # modify the reward
    #     x, x_dot, theta, theta_dot = s_
    #     r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    #     r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    #     r = r1 + r2
    #
    #     dqn.store_transition(s, a, r, s_)
    #
    #     ep_r += r
    #     if dqn.memory_counter > MEMORY_CAPACITY:
    #         dqn.learn()
    #         if done:
    #             print('Ep: ', i_episode,
    #                   '| Ep_r: ', round(ep_r, 2))
    #
    #     if episode_length == 500:
    #         break
    #     s = s_


#best parameters on training data: {'gamma': 0.046225340354787536, 'lr': 0.0005742888349514149, 'epsilon': 0.988300531974929, 'epochs': 200}
#best parameters on val
#   Value:  0.98791
#   Params:
# {'gamma': 0.00805501566596888, 'lr': 0.0005490438391140124, 'epsilon': 0.934575208045286, 'epochs': 200}