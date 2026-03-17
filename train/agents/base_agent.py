from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state, deterministic=False):
        """
        Given current state, return action.
        deterministic=True for evaluation, False for training.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update agent parameters.
        SAC passes a batch from replay buffer.
        PPO passes rollout data.
        """
        pass

    @abstractmethod
    def save(self, save_path):
        """
        Save the agent.
        """
        pass

    @abstractmethod
    def load(self, load_path):
        """
        Load the agent.
        """
        pass
