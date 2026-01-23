import nn
import numpy as np

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # 动态学习率配置
        self.lr_max = 2.0    # 初始大步长
        self.lr_min = 0.4    # 后期稳定步长
        self.lr_decay_steps = 100000 # 衰减跨度
        self.total_updates = 0
        self.learning_rate = self.lr_max

        # 训练局数相关
        self.numTrainingGames = 8000
        # 批次大小
        self.batch_size = 32

        # 稳定架构
        self.w1 = nn.Parameter(state_dim, 256)
        self.b1 = nn.Parameter(1, 256)
        self.w2 = nn.Parameter(256, 256)
        self.b2 = nn.Parameter(1, 256)
        self.w3 = nn.Parameter(256, action_dim)
        self.b3 = nn.Parameter(1, action_dim)

        self.parameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def set_weights(self, layers):
        for i in range(len(self.parameters)):
            self.parameters[i].data = np.copy(layers[i].data)
            
    def forward(self, states):
        tmp1 = nn.Linear(states, self.w1)
        tmp2 = nn.AddBias(tmp1, self.b1)
        tmp3 = nn.ReLU(tmp2)
        tmp4 = nn.Linear(tmp3, self.w2)
        tmp5 = nn.AddBias(tmp4, self.b2)
        tmp6 = nn.ReLU(tmp5)
        tmp7 = nn.Linear(tmp6, self.w3)
        return nn.AddBias(tmp7, self.b3)

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        
        Q_prediction = self.forward(states)
        
        return nn.SquareLoss(Q_prediction, Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        Q_prediction = self.forward(states)
        
        return Q_prediction

    def gradient_update(self, states, Q_target):
        # 动态学习率：从 lr_max 线性衰减到 lr_min
        self.learning_rate = max(self.lr_min, self.lr_max - (self.lr_max - self.lr_min) * (self.total_updates / self.lr_decay_steps))

        loss = self.get_loss(states, Q_target)
        grad = nn.gradients(loss, self.parameters)
        for i in range(len(self.parameters)):
            grad_dir = grad[i]
            self.parameters[i].update(grad_dir, -self.learning_rate)
            
        self.total_updates += 1
