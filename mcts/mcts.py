import math
import numpy as np

class TreeNode:
    def __init__(self, prior=0.0):
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children = {}
        self.is_expanded = False

    def expand(self, action_priors):
        for a, p in action_priors.items():
            if a not in self.children:
                self.children[a] = TreeNode(prior=p)
        self.is_expanded = True

    def update(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N


class AsyncMCTS:
    def __init__(self, env, inference_server=None, net=None, c_puct=1.0):
        self.root_env = env
        self.inference_server = inference_server
        self.net = net
        self.c_puct = c_puct
        self.root = TreeNode(prior=1.0)
        self.root_legal_actions = None

    async def search(self, n_sims=100):
        for _ in range(n_sims):
            saved_state = self.root_env.save_state()
            await self._simulate(self.root_env, self.root)
            self.root_env.restore_state(saved_state)

    def get_action_probs(self, temperature=1.0):
        counts = {}
        for a, node in self.root.children.items():
            counts[a] = node.N

        if len(counts) == 0:
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

        visits = visits ** (1.0 / temperature)
        probs = visits / np.sum(visits)

        p = np.zeros(self.root_env.ACTION_DIM, dtype=np.float32)
        for a, prob in zip(actions, probs):
            p[a] = prob
        return p

    def select_action(self, temperature=1.0, deterministic=False):
        if not self.root.children:
            pass

        counts = {a: node.N for a, node in self.root.children.items()}
        if deterministic or temperature == 0:
            if not counts: return np.random.choice(self._root_legal_actions())
            best = max(counts.items(), key=lambda kv: kv[1])[0]
            return best

        probs = self.get_action_probs(temperature)
        actions = np.nonzero(probs)[0]
        if len(actions) == 0:
            legal = self._root_legal_actions()
            return np.random.choice(legal) if len(legal) > 0 else None

        chosen = np.random.choice(actions, p=probs[actions] / probs[actions].sum())
        return int(chosen)

    async def _simulate(self, env, node):
        if not node.is_expanded:
            legal_actions = env.get_legal_actions(env.dice_value)
            legal_ids = [a["id"] for a in legal_actions]

            if self.inference_server is not None:
                state_tensor = env.encode_state(env.dice_value)
                policy_logits, value = await self.inference_server.predict(state_tensor)
                
                if isinstance(policy_logits, np.ndarray):
                    logits = policy_logits
                else:
                    logits = np.array(policy_logits)

                priors = {}
                max_logit = np.max(logits) if logits.size > 0 else 0.0
                exp_logits = np.exp(logits - max_logit)
                probs_all = exp_logits / (np.sum(exp_logits) + 1e-12)
                for aid in legal_ids:
                    priors[aid] = float(probs_all[aid])
            elif self.net is not None:
                state_tensor = env.encode_state(env.dice_value)
                policy_logits, value = self.net.predict(state_tensor)
                
                if isinstance(policy_logits, np.ndarray):
                    logits = policy_logits
                else:
                    logits = np.array(policy_logits)

                priors = {}
                max_logit = np.max(logits) if logits.size > 0 else 0.0
                exp_logits = np.exp(logits - max_logit)
                probs_all = exp_logits / (np.sum(exp_logits) + 1e-12)
                for aid in legal_ids:
                    priors[aid] = float(probs_all[aid])
            else:
                value = 0.0
                priors = {aid: 1.0 / len(legal_ids) for aid in legal_ids} if legal_ids else {}

            node.expand(priors)
            return value

        sum_N = sum(child.N for child in node.children.values()) + 1e-8
        best_score = -float("inf")
        best_action = None
        best_child = None

        for a, child in node.children.items():
            u = self.c_puct * child.P * math.sqrt(sum_N) / (1.0 + child.N)
            score = child.Q + u
            if score > best_score:
                best_score = score
                best_action = a
                best_child = child

        if best_child is None:
            return 0.0

        _, reward, done, _ = env.step(best_action)

        if done:
            leaf_value = float(np.sign(reward))
            best_child.update(leaf_value)
            return leaf_value

        leaf_value = await self._simulate(env, best_child)
        leaf_value = -leaf_value
        best_child.update(leaf_value)
        return leaf_value

    def _root_legal_actions(self):
        if self.root_legal_actions is None:
            legal = self.root_env.get_legal_actions(self.root_env.dice_value)
            self.root_legal_actions = [a["id"] for a in legal]
        return self.root_legal_actions