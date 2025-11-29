import copy
import math
import numpy as np
from collections import defaultdict


class TreeNode:
    """
    Node in MCTS.
    Stores: prior P(a), visit count N, total value W, mean value Q, children.
    """
    def __init__(self, prior=0.0):
        self.P = prior          # prior probability from policy network
        self.N = 0              # visit count
        self.W = 0.0            # total value
        self.Q = 0.0            # mean value W/N
        self.children = {}      # action_id -> TreeNode
        self.is_expanded = False

    def expand(self, action_priors):
        """
        action_priors: dict {action_id: prior_prob}
        Creates child nodes for each action with the given prior.
        """
        for a, p in action_priors.items():
            if a not in self.children:
                self.children[a] = TreeNode(prior=p)
        self.is_expanded = True

    def update(self, value):
        """Backpropagate value (value from current player's perspective)."""
        self.N += 1
        self.W += value
        self.Q = self.W / self.N


class MCTS:
    """
    Monte Carlo Tree Search class.
    Usage:
        mcts = MCTS(env, net=neural_net, c_puct=1.0)
        mcts.search(n_sims=100)
        pi = mcts.get_action_probs(temperature=1.0)
        action = mcts.select_action(temperature=1.0)
    """

    def __init__(self, env, net=None, c_puct=1.0):
        """
        env: an instance of GeneralsEnv (or a snapshot). MCTS will deepcopy it.
        net: an object with method predict(state_tensor) -> (policy_logits, value_scalar)
             If net is None, uniform priors and zero value are used.
        c_puct: exploration constant
        """
        self.root_env = env
        self.net = net
        self.c_puct = c_puct

        # Root node
        self.root = TreeNode(prior=1.0)
        # Store legal action ids for the root (populated on first expand)
        self.root_legal_actions = None

    # -------------------------
    # Public API
    # -------------------------
    def search(self, n_sims=100):
        """
        Run n_sims simulations starting from the current root.
        """
        for _ in range(n_sims):
            env_copy = copy.deepcopy(self.root_env)
            self._simulate(env_copy, self.root)

    def get_action_probs(self, temperature=1.0):
        """
        Return policy vector (length ACTION_DIM) derived from root visit counts.
        temperature:
            - 0 -> argmax (deterministic)
            - >0 -> softmax over counts^(1/temp)
        """
        # collect visit counts
        counts = {}
        for a, node in self.root.children.items():
            counts[a] = node.N

        if len(counts) == 0:
            # no children (shouldn't happen if expanded), return uniform over legal actions
            legal = self._root_legal_actions()
            if len(legal) == 0:
                return np.zeros(self.root_env.ACTION_DIM, dtype=np.float32)
            p = np.zeros(self.root_env.ACTION_DIM, dtype=np.float32)
            for a in legal:
                p[a] = 1.0 / len(legal)
            return p

        actions = np.array(list(counts.keys()))
        visits = np.array([counts[a] for a in actions], dtype=np.float64)

        if temperature == 0:
            best = actions[np.argmax(visits)]
            p = np.zeros(self.root_env.ACTION_DIM, dtype=np.float32)
            p[best] = 1.0
            return p

        # apply temperature
        visits = visits ** (1.0 / temperature)
        probs = visits / np.sum(visits)

        p = np.zeros(self.root_env.ACTION_DIM, dtype=np.float32)
        for a, prob in zip(actions, probs):
            p[a] = prob
        return p

    def select_action(self, temperature=1.0, deterministic=False):
        """
        Choose an action from root according to visit counts and temperature.
        If deterministic=True, choose argmax regardless of temperature.
        """
        if not self.root.children:
            # ensure root expanded
            self._expand_root()

        counts = {a: node.N for a, node in self.root.children.items()}
        if deterministic or temperature == 0:
            # argmax
            best = max(counts.items(), key=lambda kv: kv[1])[0]
            return best

        # sample according to probabilities
        probs = self.get_action_probs(temperature)
        actions = np.nonzero(probs)[0]
        if len(actions) == 0:
            # fallback uniform over legal actions
            legal = self._root_legal_actions()
            return np.random.choice(legal) if len(legal) > 0 else None

        chosen = np.random.choice(actions, p=probs[actions] / probs[actions].sum())
        return int(chosen)

    # -------------------------
    # Internal helpers
    # -------------------------
    def _simulate(self, env, node):
        """
        One MCTS simulation:
            selection -> expansion -> evaluation -> backpropagation
        env: a deepcopy of the environment at the node's state
        node: current TreeNode
        """

        # 1) If node is not expanded, expand it and evaluate with network
        if not node.is_expanded:
            legal_actions = env.get_legal_actions(env.dice_value)
            legal_ids = [a["id"] for a in legal_actions]

            # obtain priors and value from net (or fallback)
            if self.net is not None:
                # net.predict expects the encoded state tensor
                state_tensor = env.encode_state(env.dice_value)
                policy_logits, value = self.net.predict(state_tensor)
                # policy_logits assumed as numpy array of shape (ACTION_DIM,) (logits)
                # convert to probabilities and restrict to legal actions
                if isinstance(policy_logits, np.ndarray):
                    logits = policy_logits
                else:
                    logits = np.array(policy_logits)

                priors = {}
                # softmax over logits (numerical stable)
                max_logit = np.max(logits) if logits.size > 0 else 0.0
                exp_logits = np.exp(logits - max_logit)
                probs_all = exp_logits / (np.sum(exp_logits) + 1e-12)
                for aid in legal_ids:
                    priors[aid] = float(probs_all[aid])
            else:
                # uniform priors & zero value
                value = 0.0
                priors = {aid: 1.0 / len(legal_ids) for aid in legal_ids} if legal_ids else {}

            # expand node with priors
            node.expand(priors)

            # return value for backpropagation (value from current player's perspective)
            return value

        # 2) Selection: choose child with max UCB
        # compute total visit count sum_N
        sum_N = sum(child.N for child in node.children.values()) + 1e-8

        best_score = -float("inf")
        best_action = None
        best_child = None

        for a, child in node.children.items():
            # UCB = Q + c_puct * P * sqrt(sum_N) / (1 + N)
            u = self.c_puct * child.P * math.sqrt(sum_N) / (1.0 + child.N)
            score = child.Q + u
            if score > best_score:
                best_score = score
                best_action = a
                best_child = child

        if best_child is None:
            # no children (terminal?). return 0 fallback
            return 0.0

        # Apply action 'best_action' on env
        # Use env.step(best_action) - step returns (next_state, reward, done, info)
        _, reward, done, _ = env.step(best_action)

        if done:
            # Terminal node: compute leaf value from env perspective
            # reward in our env is from the player who acted; but MCTS convention expects
            # a value from the viewpoint of the player who is to move at the node.
            # We'll treat reward as:
            #   if reward > 0 -> current player at node caused a win -> value = +1
            #   if reward == 0 -> draw/ongoing -> value = 0
            #   if reward < 0 -> loss -> -1
            leaf_value = float(np.sign(reward))
            # backpropagate leaf_value
            self._backpropagate(best_child, leaf_value)
            return leaf_value

        # otherwise continue simulation recursively from child node with updated env
        leaf_value = self._simulate(env, best_child)

        # after recursion returns value (from perspective of the player to move at child),
        # we need to flip sign because value is zero-sum alternating-player viewpoint
        leaf_value = -leaf_value

        # update child stats with leaf_value
        best_child.update(leaf_value)

        return leaf_value

    def _backpropagate(self, node, value):
        """
        Backpropagate value up from node to root. Value is from perspective of the player to move at 'node'.
        We update each visited node by alternating sign when moving up the tree.
        """
        # We don't keep parent pointers; instead we'll rely on the fact
        # that _simulate updates nodes during recursion. This function is used
        # only for terminal immediate returns in this implementation.
        node.update(value)

    def _expand_root(self):
        """
        Expand root if not expanded, using current root_env state.
        """
        if self.root.is_expanded:
            return
        legal_actions = self.root_env.get_legal_actions(self.root_env.dice_value)
        legal_ids = [a["id"] for a in legal_actions]

        if self.net is not None:
            st = self.root_env.encode_state(self.root_env.dice_value)
            logits, _ = self.net.predict(st)
            logits = np.array(logits)
            max_logit = np.max(logits) if logits.size > 0 else 0.0
            exp_logits = np.exp(logits - max_logit)
            probs_all = exp_logits / (np.sum(exp_logits) + 1e-12)
            priors = {aid: float(probs_all[aid]) for aid in legal_ids}
        else:
            priors = {aid: 1.0 / len(legal_ids) for aid in legal_ids} if legal_ids else {}

        self.root.expand(priors)
        self.root_legal_actions = legal_ids

    def _root_legal_actions(self):
        if self.root_legal_actions is None:
            legal = self.root_env.get_legal_actions(self.root_env.dice_value)
            self.root_legal_actions = [a["id"] for a in legal]
        return self.root_legal_actions