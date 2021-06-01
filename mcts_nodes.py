from utils import*
import numpy as np
from global_params import*
from tqdm import tqdm, trange


class Node:
    """ a tree node for MCTS """

    # metadata:
    parent = None  # parent Node
    value_sum = 0.  # sum of state values from all visits (numerator)
    times_visited = 0  # counter of visits (denominator)
    network_value_sum = 0  # W in the article

    def __init__(self, parent, action, env):
        """
        Creates and empty node with no children.
        Does so by commiting an action and recording outcome.

        :param parent: parent Node
        :param action: action to commit from parent Node

        """

        self.parent = parent
        self.action = action
        self.children = set()  # set of child nodes
        # get action outcome and save it
        res = env.get_result(parent.snapshot, action)
        self.snapshot, self.observation, self.immediate_reward, self.is_done, _ = res
        self.env = env
        state_pics = get_pics(self, env)
        self.state = format_pics(state_pics)

        # self.prior_probs = sess.run([policy_output_layer], feed_dict={state_t: state}) #Pa in the article
        if self.parent is not None:
            self.prob = self.parent.prior_probs[action]
        else:
            self.prob = 1

        self.prior_probs = None
        self.action_value = 0  # Q in the article

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def get_mean_value(self):
        return self.value_sum / self.times_visited if self.times_visited != 0 else 0

    def ucb_score(self, scale=1 / np.sqrt(2), max_value=1e100):
        """
        Computes ucb1 upper bound using current value and visit counts for node and it's parent.

        :param scale: Multiplies upper bound by that. From hoeffding inequality, assumes reward range to be [0,scale].
        :param max_value: a value that represents infinity (for unvisited nodes)

        """

        if self.times_visited == 0:
            return max_value

        # compute ucb-1 additive component (to be added to mean value)
        # hint: you can use self.parent.times_visited for N times node was considered,
        # and self.times_visited for n times it was visited

        if self.parent is not None:
            parent_visits = self.parent.times_visited
        else:
            parent_visits = 0
        U = np.sqrt(2 * np.log(self.parent.times_visited) / self.times_visited)

        return self.get_mean_value() + scale * U

    def poly_upper_conf_score(self, c_puct=10):  # selecting by the PUCT algorithm

        # U = c_puct * self.prob * np.sqrt(np.sum([child.times_visited for child in self.children])) / (
        #             1 + self.times_visited)

        if self.parent is None:
            U = 0
        else:
            child_visits = [child.times_visited for child in self.parent.children]
            # U = c_puct * self.prob * np.sqrt(self.parent.times_visited)/(1 + self.times_visited)
            U = c_puct * self.prob * np.sqrt(np.sum(child_visits)) / (1 + self.times_visited)
        return self.action_value + U

    # MCTS steps

    def select_best_leaf(self):
        """
        Picks the leaf with highest priority to expand
        Does so by recursively picking nodes with best UCB-1 score until it reaches the leaf.

        """
        if self.is_leaf():
            return self

        # best_child = <YOUR CODE: select best child node in terms of node.ucb_score()>
        children = list(self.children)
        children_puct = [node.poly_upper_conf_score() for node in children]
        ii = np.argmax(children_puct)
        best_child = children[ii]  # <select best child node in terms of node.ucb_score()>

        return best_child.select_best_leaf()

    def expand(self, sess, policy_output_layer, value_output_layer, state_t):
        """
        Expands the current node by creating all possible child nodes.
        Then returns one of those children.

        input:
            sess                - tensorflow v1 session
            policy_output_layer - a tensor of the policy output layer
            value_output_layer  - a tensorf of the value output layer
            state_t             - a placeholder of the current None state

        all inputs are used for running the current session with the current network weights
        """

        assert not self.is_done, "can't expand from terminal state"

        self.prior_probs, self.network_value_sum = sess.run([policy_output_layer, value_output_layer],
                                                            feed_dict={state_t: self.state})

        self.prior_probs = self.prior_probs[0]
        self.prior_probs = (1 - dir_noise_weight) * self.prior_probs + dir_noise_weight * np.random.dirichlet(
            alpha=dirichlet_alpha)
        self.network_value_sum = self.network_value_sum[0]
        # when we expand we automatically need to update visit count, since the propagation will be from the parent of the expanded node and up.
        self.times_visited += 1
        self.action_value = self.network_value_sum / self.times_visited

        for action in range(n_actions):
            self.children.add(Node(self, action, self.env))

        return self.select_best_leaf()

    def rollout(self, t_max=10 ** 4):
        """
        Play the game from this state to the end (done) or for t_max steps.

        On each step, pick action at random (hint: env.action_space.sample()).

        Compute sum of rewards from current state till
        Note 1: use env.action_space.sample() for random action
        Note 2: if node is terminal (self.is_done is True), just return 0

        """

        # set env into the appropriate state
        self.env.restore_snapshot(self.snapshot)
        obs = self.observation
        is_done = self.is_done
        rollout_reward = 0
        if is_done:
            return 0
        for t in range(t_max):
            l1 = self.env.unwrapped.ale.lives()
            #env.spec.max_episode_steps += 10000000000000
            _, r, is_done, _ = self.env.step(self.env.action_space.sample())
            l2 = self.env.unwrapped.ale.lives()
            if l2 < l1:
                r -= 3
            # env.render()
            #env.spec.max_episode_steps += 10000000000000
            rollout_reward += r
            if is_done:
                break

        return rollout_reward

    def propagate(self, child_network_value):
        """
        Uses child value (sum of rewards) to update parents recursively.
        """
        # compute node value
        self.network_value_sum += child_network_value
        self.times_visited += 1
        self.action_value = self.network_value_sum / self.times_visited

        # propagate upwards
        if not self.is_root():
            self.parent.propagate(child_network_value)  # TODO: Figure out if child_value or network value sum

    def safe_delete(self):
        """safe delete to prevent memory leak in some python versions"""
        del self.parent
        for child in self.children:
            child.safe_delete()
            del child

    def pop_pic(self):
        if len(self.children) == 0:
            self.env.pics.pop(self.snapshot)
        else:
            children = self.children
            self.env.pics.pop(self.snapshot)
            for child in children:
                child.pop_pic()
                del child.state

    def find_root(self):
        if self.parent is None:
            if hasattr(self, 'state'):
                del self.state
            return self
        else:
            if hasattr(self, 'state'):
                del self.state
            for child in self.children:
                if hasattr(child, 'state'):
                    del child.state
            return self.parent.find_root()


class Root(Node):
    def __init__(self, snapshot, observation, env):
        """
        creates special node that acts like tree root
        :snapshot: snapshot (from env.get_snapshot) to start planning from
        :observation: last environment observation
        """

        self.parent = self.action = None
        self.children = set()  # set of child nodes

        self.value_sum = 0.  # sum of state values from all visits (numerator)
        self.times_visited = 0  # counter of visits (denominator)
        self.network_value_sum = 0
        # root: load snapshot and observation
        self.snapshot = snapshot
        self.observation = observation
        self.is_done = False
        state_pics = get_pics(self, env)
        # print(state_pics.shape)
        self.state = format_pics(state_pics)
        self.env = env

    @staticmethod
    def from_node(node):
        """initializes node as root"""
        root = Root(node.snapshot, node.observation, node.env)
        # copy data
        copied_fields = ["value_sum", "times_visited", "children", "is_done", "prior_probs", "network_value_sum",
                         "state"]
        for field in copied_fields:
            setattr(root, field, getattr(node, field))

        if node.parent is not None:
            for child in node.parent.children:
                if child == node:
                    continue
                child.pop_pic()
        return root

    def play_move(self, num_move):
        if num_move < 30:
            temp = move_temp_initial
        else:
            temp = move_temp
        p = np.zeros((n_actions))
        # count = 0
        total_n = np.sum([child.times_visited ** (1 / temp) for child in self.children])
        c_list = {}
        c_val_list = {}
        if total_n == 0:
            print('default action')  # if we got here it means that every child has is_done marked as true!!!
            cc = list(self.children)
            return cc[0].action, cc[0], p, cc[0].action_value  # p and ac_val are nonesense but it doesn't matter since the child we're choosing is is_done

        for child in self.children:
            p[child.action] = (child.times_visited ** (1 / temp)) / total_n
            c_list[child.action] = child
            c_val_list[child.action] = child.action_value

        move = np.random.choice(np.arange(n_actions), p=p)
        if np.sum(p) != 1:
            print("sum: " + str(np.sum(p)))
        # similarity should be between policy pobabilities and the final tree probability. NOT to the one hot
        # rep of the chosen action, but the entire probability computed = p
        return move, c_list[move], p, c_val_list[move]


def plan_mcts(root, sess, policy_output_layer, value_output_layer, state_t, n_iters=1600):
    """
    builds tree with monte-carlo tree search for n_iters iterations
    :param root: tree node to plan from
    :param sess: current tensorflow session
    :param policy_output_layer: tensor for policy vector
    :param value_output_layer: tensor for the value of a specific state
    :param state_t: placeholder for the current state
    :param n_iters: how many select-expand-simulate-propagete loops to make
    """
    for _ in tqdm(range(n_iters)):
        node = root.select_best_leaf()
        if node.is_done and node.parent is None:
            return

        if node.is_done:
            node.parent.propagate(0)

        else:  # node is not terminal
            leaf = node.expand(sess, policy_output_layer, value_output_layer, state_t)
            v_value = node.network_value_sum
            if node.parent is not None:
                node.parent.propagate(v_value)
            root.env.restore_snapshot(leaf.snapshot)