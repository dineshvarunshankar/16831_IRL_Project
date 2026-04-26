"""
Base Algorithm Interface

All custom RL algorithms (PPO, SAC, TDMPC2) implement this interface
for consistent usage across the training pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch


class BaseAlgorithm(ABC):
    """
    Base class for all RL algorithms.
    
    All algorithms must implement this interface to work with the
    Unitree RL mjlab training pipeline.
    """
    
    @abstractmethod
    def select_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Select actions for batched observations.
        
        Args:
            obs: Observations tensor of shape [num_envs, obs_dim]
            deterministic: If True, return deterministic actions (no exploration)
        
        Returns:
            actions: Actions tensor of shape [num_envs, act_dim]
        """
        pass
    
    @abstractmethod
    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update algorithm from batched transitions.
        
        Args:
            obs: Observations [num_envs, obs_dim]
            actions: Actions [num_envs, act_dim]
            rewards: Rewards [num_envs]
            next_obs: Next observations [num_envs, obs_dim]
            dones: Done flags [num_envs]
        
        Returns:
            metrics: Dictionary of training metrics (losses, etc.)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint file
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current training metrics.
        
        Returns:
            metrics: Dictionary of current metrics
        """
        return {}

    # Optional off-policy extension hooks.
    # Algorithms such as SAC / TD-MPC2 may use internal replay and call update
    # without passing a batch through the public update(...) method.
    def store_transition(self, **kwargs) -> None:
        """
        Store one batched transition into algorithm-managed replay memory.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement store_transition()"
        )

    def update_from_replay(self) -> Dict[str, float]:
        """
        Run one update pass using algorithm-managed replay memory.
        """
        return {}
