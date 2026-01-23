import nn
import model
from qlearningAgents import PacmanQAgent
from backend import ReplayMemory
import layout
import copy

import numpy as np

class PacmanDeepQAgent(PacmanQAgent):
    def __init__(self, layout_input="smallGrid", target_update_rate=300, doubleQ=True, **args):
        PacmanQAgent.__init__(self, **args)
        self.model = None
        self.target_model = None
        self.target_update_rate = target_update_rate
        self.update_amount = 0
        self.epsilon_explore = 1.0
        self.epsilon0 = 0.05
        self.epsilon = self.epsilon_explore
        self.discount = 0.95 # 稍微调低折扣率，关注短期食物
        self.update_frequency = 1
        self.replay_memory = ReplayMemory(50000)
        self.min_transitions_before_training = 1000
        self.td_error_clipping = 1.0

        # Initialize Q networks:
        if isinstance(layout_input, str):
            layout_instantiated = layout.getLayout(layout_input)
        else:
            layout_instantiated = layout_input
        self.state_dim = self.get_state_dim(layout_instantiated)
        self.initialize_q_networks(self.state_dim)

        self.doubleQ = doubleQ
        if self.target_update_rate == -1:
            self.target_update_rate = 500
        
        # 修复同步问题，确保代理知晓训练局数
        self.numTraining = self.model.numTrainingGames
        
        self.all_actions = ['North', 'South', 'East', 'West', 'Stop']

    def get_state_dim(self, layout):
        return 3 * layout.width * layout.height

    def get_features(self, state):
        food_grid = state.getFood()
        width, height = food_grid.width, food_grid.height
        pac_grid = np.zeros((width, height))        gradient * (inputs[0] - inputs[1]) / inputs[0].size
        px, py = state.getPacmanPosition()
        pac_grid[px][py] = 1.0
        ghost_grid = np.zeros((width, height))
        for gx, gy in state.getGhostPositions():
            ghost_grid[int(gx)][int(gy)] = 1.0
        f_data = np.array(food_grid.data).astype(float)
        return np.concatenate([pac_grid.flatten(), ghost_grid.flatten(), f_data.flatten()])

    def initialize_q_networks(self, state_dim, action_dim=5):
        import model
        self.model = model.DeepQNetwork(state_dim, action_dim)
        self.target_model = model.DeepQNetwork(state_dim, action_dim)
        self.target_model.set_weights(self.model.parameters)

    def getQValue(self, state, action):
        """
          Should return Q(state,action) as predicted by self.model
        """
        feats = self.get_features(state)
        action_index = self.all_actions.index(action)
        state_node = nn.Constant(np.array([feats]).astype("float64"))
        return self.model.run(state_node).data[0][action_index]


    def shape_reward(self, reward):
        # 稳定的奖励塑造
        if reward > 100: return 10.0   # Win
        if reward < -100: return -10.0 # Loss
        if reward > 5: return 2.0      # Food
        return -0.1                    # Move


    def compute_q_targets(self, minibatch, network = None, target_network=None, doubleQ=False):
        """Prepare minibatches
        Args:
            minibatch (List[Transition]): Minibatch of `Transition`
        Returns:
            float: Loss value
        """
        if network is None:
            network = self.model
        if target_network is None:
            target_network = self.target_model
        states = np.vstack([x.state for x in minibatch])
        states = nn.Constant(states)
        actions = np.array([x.action for x in minibatch])
        rewards = np.array([x.reward for x in minibatch])
        next_states = np.vstack([x.next_state for x in minibatch])
        next_states = nn.Constant(next_states)
        done = np.array([x.done for x in minibatch])

        Q_predict = network.run(states).data
        Q_target = np.copy(Q_predict)

        replace_indices = np.arange(actions.shape[0])
        
        if doubleQ:
            action_indices = np.argmax(network.run(next_states).data, axis=1)
        else:
            action_indices = np.argmax(target_network.run(next_states).data, axis=1)

        target = rewards + (1 - done) * self.discount * target_network.run(next_states).data[replace_indices, action_indices]

        Q_target[replace_indices, actions] = target

        if self.td_error_clipping is not None:
            Q_target = Q_predict + np.clip(
                     Q_target - Q_predict, -self.td_error_clipping, self.td_error_clipping)

        return Q_target

    def update(self, state, action, nextState, reward):
        action_index = self.all_actions.index(action)
        done = nextState.isLose() or nextState.isWin()
        reward = self.shape_reward(reward)

        state_feats = self.get_features(state)
        next_state_feats = self.get_features(nextState)

        self.replay_memory.push(state_feats, action_index, reward, next_state_feats, done)

        # 衰减周期 - 调大以确保足够的训练时间
        decay_steps = 50000
        if len(self.replay_memory) < self.min_transitions_before_training:
            self.epsilon = self.epsilon_explore
        else:
            self.epsilon = max(self.epsilon0, self.epsilon_explore - (self.epsilon_explore - self.epsilon0) * (self.update_amount - self.min_transitions_before_training) / decay_steps)

        if len(self.replay_memory) > self.min_transitions_before_training and self.update_amount % self.update_frequency == 0:
            minibatch = self.replay_memory.pop(self.model.batch_size)
            states = np.vstack([x.state for x in minibatch])
            states = nn.Constant(states.astype("float64"))
            Q_target1 = self.compute_q_targets(minibatch, self.model, self.target_model, doubleQ=self.doubleQ)
            Q_target1 = nn.Constant(Q_target1.astype("float64"))

            self.model.gradient_update(states, Q_target1)

        if self.target_update_rate > 0 and self.update_amount % self.target_update_rate == 0:
            self.target_model.set_weights(self.model.parameters)

        self.update_amount += 1

    def final(self, state):
        """Called at the end of each game."""
        PacmanQAgent.final(self, state)
