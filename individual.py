from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, TYPE_CHECKING
import itertools
import numpy as np
from memory_system import MemoryBank, MemoryConfig
from program import Program
from instruction_set import InstructionSet
from operators import GeneticOperators

if TYPE_CHECKING:
    from evaluator import FitnessEvaluator

# Global counter for Individual IDs
_individual_counter = itertools.count()

@dataclass
class Individual:
    """A candidate solution with program and evolvable constants"""
    program: Program
    memory: MemoryBank
    id: int = field(default_factory=lambda: next(_individual_counter))
    fitness: Optional[float] = None
    age: int = 0
    parent_ids: Tuple[int, ...] = ()
    
    def evaluate(self, evaluator: 'FitnessEvaluator') -> float:
        """Evaluate fitness using given evaluator"""
        if self.fitness is None:
            self.fitness = evaluator.evaluate(self)
        return self.fitness
    
    def invalidate_fitness(self):
        """Mark fitness as needing re-evaluation"""
        self.fitness = None
    
    def copy(self, new_id: bool = True) -> 'Individual':
        """
        Deep copy of individual.
        
        Args:
            new_id: If True, assigns new ID (for offspring).
                    If False, keeps same ID (for cloning/caching).
        """
        return Individual(
            program=self.program.copy(),
            memory=self.memory.copy(),
            id=next(_individual_counter) if new_id else self.id,
            fitness=self.fitness,
            age=self.age,
            parent_ids=self.parent_ids
        )
    
    def create_offspring(self, parent_ids: Tuple[int, ...]) -> 'Individual':
        """Create offspring with this individual's program but new ID and parents"""
        offspring = self.copy(new_id=True)
        offspring.parent_ids = parent_ids
        offspring.age = 0
        offspring.invalidate_fitness()
        return offspring
    
    def get_effective_length(self, output_registers) -> int:
        """Get effective program length (for parsimony)"""
        return self.program.get_effective_length(output_registers)
    
    def get_constants(self) -> Dict:
        """Get evolvable constants (for analysis)"""
        return self.memory.get_constants()

    @classmethod
    def random(
        cls,
        instruction_set: InstructionSet,
        memory_config: MemoryConfig,
        program_length: int,
        rng: Optional[np.random.Generator] = None,
        mutate_constants: bool = True,
    ) -> 'Individual':
        rng = rng or np.random.default_rng()
        program = instruction_set.generate_random_program(program_length, rng)
        memory = MemoryBank(
            n_scalar=memory_config.n_scalar,
            n_vector=memory_config.n_vector,
            n_matrix=memory_config.n_matrix,
            n_obs_scalar=memory_config.n_obs_scalar,
            n_obs_vector=memory_config.n_obs_vector,
            n_obs_matrix=memory_config.n_obs_matrix,
            vector_size=memory_config.vector_size,
            matrix_shape=memory_config.matrix_shape,
            init_scalar_range=memory_config.init_scalar_range,
            init_vector_range=memory_config.init_vector_range,
            init_matrix_range=memory_config.init_matrix_range,
        )
        if mutate_constants:
            GeneticOperators(instruction_set, rng).mutate_constants(memory, rng)
        return cls(program=program, memory=memory)
    
    def __repr__(self) -> str:
        fitness_str = f"{self.fitness:.2f}" if self.fitness is not None else "None"
        return f"Individual(id={self.id}, fitness={fitness_str}, len={len(self.program)}, age={self.age})"


if __name__ == "__main__":
    from operation import ALL_OPS

    rng = np.random.default_rng(123)

    memory_cfg = MemoryConfig(
        n_scalar=4,
        n_vector=2,
        n_matrix=1,
        n_obs_scalar=2,
        n_obs_vector=1,
        n_obs_matrix=0,
        vector_size=3,
        matrix_shape=(2, 2),
    )

    from instruction_set import InstructionSet
    from memory_system import MemoryBank

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

    instr_set = InstructionSet([op() for op in ALL_OPS], template_memory)

    ind = Individual.random(
        instruction_set=instr_set,
        memory_config=memory_cfg,
        program_length=5,
        rng=rng,
        mutate_constants=True,
    )

    print("Generated Individual:", ind)
    print("Program instructions:")
    for idx, instr in enumerate(ind.program.instructions):
        print(f"  {idx}: {instr}")

    print("\nInitial scalar registers:", ind.memory.scalars)
    print("Initial vector registers:\n", ind.memory.vectors)
    print("Initial matrix registers:\n", ind.memory.matrices)