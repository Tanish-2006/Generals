try:
    from env.generals_env import GeneralsEnv
except ImportError:
    pass


class GameEnvironment:
    """
    Wrapper around GeneralsEnv to expose cleaner API for the app
    """

    def __init__(self):
        self.env = GeneralsEnv()

    def reset(self):
        self.env.reset()
        return self.get_state_data()

    def step(self, action_id: int):
        state, reward, done, info = self.env.step(action_id)
        state_data = self.get_state_data()

        # Add messages based on info
        msgs = []
        if info.get("stalemate_rerolls", 0) > 0:
            rerolls = info["stalemate_rerolls"]
            msgs.append(f"Stalemate! Auto-rerolled {rerolls} times.")

        state_data["message"] = " ".join(msgs) if msgs else None

        return state_data, done, self.env.winner

    def get_state_data(self):
        """
        Returns serializable state data
        """
        legal = self.env.get_legal_actions()
        legal_ids = [a["id"] for a in legal]

        # Map player ID to A/B
        player_str = "A" if self.env.current_player == 1 else "B"
        attacker_str = "A" if self.env.attacker_player == 1 else "B"
        defender_str = "A" if self.env.defender_player == 1 else "B"

        return {
            "board": self.env.board.tolist(),
            "current_player": player_str,
            "legal_actions": legal_ids,
            "dice_value": int(self.env.dice_value),
            "general_hits": int(self.env.general_hits),
            "turn": int(self.env.turn),
            "attacker": attacker_str,
            "defender": defender_str,
            "message": None,
            "last_move_description": None,
        }

    @property
    def raw_env(self):
        return self.env
