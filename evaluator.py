"""Fitness evaluation helpers for Linear Genetic Programming."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, List, Tuple

import numpy as np


import gymnasium as gym  # type: ignore[import]


from memory_system import MemoryType

from individual import Individual  


class FitnessEvaluator(ABC):
    """Base class for evaluating individuals.

    Subclasses implement `_evaluate_episode` to define how a single
    evaluation episode is scored. The public `evaluate` method averages
    across episodes and returns a scalar fitness.
    """

    def __init__(
        self,
        episodes: int = 1,
        rng: Optional[np.random.Generator] = None,
        output_registers: Optional[List[Tuple[MemoryType, int]]] = None,
    ) -> None:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        self.episodes = episodes
        self.rng = rng or np.random.default_rng()
        self.output_registers: List[Tuple[MemoryType, int]] = output_registers or []

    def evaluate(self, individual: 'Individual') -> float:
        rewards = [
            float(self._evaluate_episode(individual, episode_idx))
            for episode_idx in range(self.episodes)
        ]
        return float(np.mean(rewards)) if rewards else 0.0

    @abstractmethod
    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        """Return the reward obtained by the individual in a single episode."""

class SymbolicRegressionEvaluator(FitnessEvaluator):
    """Simple evaluator for testing: fit z = x + y.

    - Two scalar observations (x, y) are provided in obs registers [-1], [-2].
    - The program's output is read from working scalar register 0.
    - Fitness is negative absolute error so that higher is better.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(episodes=5, rng=rng, output_registers=[(MemoryType.SCALAR, 0)])

    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        memory = individual.memory.copy()

        x, y = self.rng.uniform(-1.0, 1.0, size=2)
        memory.load_observation({'scalar': [x, y]})

        individual.program.execute(memory)

        predicted = memory.read_scalar(0)
        target = x + y
        error = abs(predicted - target)
        return -error

class CartPoleEvaluator(FitnessEvaluator):
    """Evaluate a policy on CartPole using scalar observations and output register.

    Assumptions:
        - Observation registers [-1], [-2], [-3], [-4] store cartpole state.
        - Working scalars include register `output_register` which encodes the action.
        - The program writes to the designated output register after execution.
    """

    def __init__(
        self,
        env_id: str = "CartPole-v1",
        episodes: int = 10,
        max_steps: int = 500,
        output_register: int = 7,
        render_mode: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if gym is None:
            raise ImportError("gymnasium is required for CartPoleEvaluator")
        super().__init__(
            episodes=episodes,
            rng=rng,
            output_registers=[(MemoryType.SCALAR, output_register)],
        )
        self.env = gym.make(env_id, render_mode=render_mode)
        self.max_steps = max_steps
        self.output_register = output_register

    def close(self) -> None:
        if hasattr(self, "env") and self.env is not None:
            self.env.close()

    def __del__(self):  # pragma: no cover
        self.close()

    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        
        observation, _ = self.env.reset()
        observation = np.asarray(observation, dtype=np.float32)

        memory = individual.memory.copy()
        total_reward = 0.0

        for _ in range(self.max_steps):
            memory.load_observation({'scalar': observation.tolist()})

            individual.program.execute(memory)

            action_value = memory.read_scalar(self.output_register)
            action = 1 if action_value >= 0.0 else 0

            observation, reward, terminated, truncated, _ = self.env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            total_reward += reward
            if terminated or truncated:
                break

        return float(total_reward)


class FlappyBirdEvaluator(FitnessEvaluator):
    """Placeholder for future visual RL evaluation."""

    def __init__(self, episodes: int = 1, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(episodes=episodes, rng=rng)

    def _evaluate_episode(self, individual: 'Individual', episode_idx: int) -> float:
        raise NotImplementedError("FlappyBirdEvaluator is not yet implemented")



if __name__ == "__main__":
    from memory_system import MemoryConfig, MemoryBank
    from instruction_set import InstructionSet
    from operation import ALL_OPS, SCALAR_OPS
    from individual import Individual  # type: ignore[import]

    rng = np.random.default_rng(0)

    memory_cfg = MemoryConfig(
        n_scalar=8,
        n_vector=0,
        n_matrix=0,
        n_obs_scalar=4,
        n_obs_vector=0,
        n_obs_matrix=0,
        vector_size=1,
        matrix_shape=(1, 1),
    )

    template_memory = MemoryBank(
        n_scalar=memory_cfg.n_scalar,
        n_vector=memory_cfg.n_vector,
        n_matrix=memory_cfg.n_matrix,
        n_obs_scalar=memory_cfg.n_obs_scalar,
        n_obs_vector=memory_cfg.n_obs_vector,
        n_obs_matrix=memory_cfg.n_obs_matrix,
        vector_size=memory_cfg.vector_size,
        matrix_shape=memory_cfg.matrix_shape,
    )

    instr_set = InstructionSet([op() for op in SCALAR_OPS], template_memory)
    individual = Individual.random(instr_set, memory_cfg, program_length=6, rng=rng)

    # evaluator = SymbolicRegressionEvaluator(rng)
    # fitness = evaluator.evaluate(individual)
    # print("Symbolic regression fitness:", fitness)

    if gym is not None:
        cartpole_eval = CartPoleEvaluator(episodes=2, rng=rng)
        # Ensure individual has properly sized memory for cartpole; reuse existing for demo.
        cartpole_fitness = cartpole_eval.evaluate(individual)
        print("CartPole fitness:", cartpole_fitness)