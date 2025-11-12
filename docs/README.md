# LGP Vision System Documentation

## Overview

This documentation provides comprehensive information about the Linear Genetic Programming (LGP) Vision system. The system is designed for evolutionary computation using typed memory registers (scalars, vectors, matrices) and supports observation (read-only) and working (read-write) memory regions.

## Documentation Index

### Core Modules

1. **[Memory System](memory_system.md)**
   - `MemoryConfig` dataclass for describing memory layouts
   - `MemoryBank` class: Manages typed memory registers
   - `MemoryType` enum: Defines register types (SCALAR, VECTOR, MATRIX)
   - Observation vs working register distinction
   - Memory access methods and index wrapping

2. **[Operations](operation.md)**
   - `Operation` abstract base class
   - 33 operation implementations (scalar, vector, matrix, cross-type)
   - Operation registry and helper functions
   - Type safety and differentiability information

3. **[Instructions](instruction.md)**
   - `Instruction` class: Single instruction execution unit
   - Type-safe instruction creation
   - Memory access patterns (observation vs working registers)
   - Dependency analysis methods

4. **[Programs](program.md)**
   - `Program` class: Sequence of instructions
   - Program execution
   - Intron removal system (dead code elimination)
   - Backward dependency analysis
   - Program statistics and metrics

5. **[Instruction Set](instruction_set.md)**
   - `InstructionSet` class: Random instruction generation
   - Type-safe program generation
   - Memory configuration management
   - Random program generation

6. **[Individual](individual.md)**
   - `Individual` class: wraps program, memory, and fitness metadata
   - `Individual.random()` helper for spawning new candidates
   - Copy/offspring utilities for lineage tracking

7. **[Population](population.md)**
   - `PopulationConfig` and `Population` classes
   - Tournament selection, elitism, statistics, and evaluation helpers

8. **[Genetic Operators](operators.md)**
   - Program-level micro and macro mutations
   - Constant mutation (`mutate_constants`) with vectorised speedups
   - One-/two-point crossover and program-level helpers

9. **[Evaluator](evaluator.md)**
   - `FitnessEvaluator` base class with multi-episode support
   - `SymbolicRegressionEvaluator` example
   - `CartPoleEvaluator` integrating gymnasium for RL
   - Placeholder `FlappyBirdEvaluator` for future RL integration

10. **[Evolution Engine](evolution_engine.md)**
    - `EvolutionConfig` for loop parameters
    - `EvolutionEngine` orchestration of evaluation, variation, and replacement

11. **[Agent](agent.md)** *(future work)*
    - Planned environment interaction layer for RL tasks

---

## System Architecture

### Data Flow

```
MemoryBank (state)
    ↓
Instructions (operations)
    ↓
Program (sequence)
    ↓
Execution → Output Registers
```

### Memory Model

```
MemoryBank
├── Observation Registers (read-only)
│   ├── Scalar observations
│   ├── Vector observations
│   └── Matrix observations
└── Working Registers (read-write)
    ├── Scalar working registers
    ├── Vector working registers
    └── Matrix working registers
```

### Instruction Model

```
Instruction
├── Operation (what to do)
├── Source Registers (where to read from)
│   ├── Observation registers (negative indices)
│   └── Working registers (non-negative indices)
└── Destination Register (where to write)
    └── Working register only (non-negative index)
```

---

## Key Concepts

### Type System

The system uses a strict type system:
- **SCALAR**: Single floating-point value
- **VECTOR**: 1D array of floating-point values
- **MATRIX**: 2D array of floating-point values

Operations are type-safe: input and output types must match register types.

### Observation vs Working Registers

- **Observation Registers**: Read-only, updated externally (environment inputs)
- **Working Registers**: Read-write, used for program computation

### Index Wrapping

All register indices automatically wrap using modulo arithmetic:
- Prevents `IndexError` exceptions
- Allows flexible program generation
- Handles out-of-bounds indices gracefully

### Intron Removal

Introns are instructions that don't affect program output (dead code). The system:
- Identifies effective instructions using backward dependency analysis
- Removes introns to create optimized programs
- Maintains instruction order in optimized programs

---

## Quick Start

### Creating a Simple Program

```python
from memory_system import MemoryBank, MemoryType
from instruction import Instruction
from operation import ScalarAddOp, ScalarMulOp
from program import Program

# Create memory bank
memory = MemoryBank(
    n_scalar=10, n_vector=0, n_matrix=0,
    n_obs_scalar=2, n_obs_vector=0, n_obs_matrix=0,
    vector_size=0, matrix_shape=(0, 0)
)

# Load observations
memory.load_observation({'scalar': [2.0, 3.0]})

# Create instructions
instructions = [
    Instruction(ScalarAddOp(), MemoryType.SCALAR, 0,
               [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),
    Instruction(ScalarMulOp(), MemoryType.SCALAR, 1,
               [MemoryType.SCALAR, MemoryType.SCALAR], [0, 0]),
    Instruction(ScalarAddOp(), MemoryType.SCALAR, 9,
               [MemoryType.SCALAR, MemoryType.SCALAR], [1, 1])
]

# Create and execute program
program = Program(instructions)
program.execute(memory)

# Read output
output = memory.read_scalar(9)
print(f"Output: {output}")
```

### Generating Random Programs

```python
from instruction_set import InstructionSet
from memory_system import MemoryBank
from operation import ALL_OPS
import numpy as np

# Setup
memory = MemoryBank(...)
operations = [op() for op in ALL_OPS]
instruct_set = InstructionSet(operations, memory)

# Generate random program
rng = np.random.default_rng(42)
program = instruct_set.generate_random_program(20, rng)
```

### Intron Removal

```python
# Find effective instructions
output_regs = [(MemoryType.SCALAR, 9)]
effective = program.find_effective_instructions(output_regs)

# Remove introns
compact_program = program.remove_introns(output_regs)

# Get statistics
intron_ratio = program.get_intron_ratio(output_regs)
print(f"Intron ratio: {intron_ratio:.1%}")
```

### Running Evolution

```python
from memory_system import MemoryConfig, MemoryBank
from instruction_set import InstructionSet
from operation import ALL_OPS
from population import Population, PopulationConfig
from evolution_engine import EvolutionEngine, EvolutionConfig
from operators import GeneticOperators
from evaluator import SymbolicRegressionEvaluator
import numpy as np

rng = np.random.default_rng(0)

memory_cfg = MemoryConfig(
    n_scalar=6, n_vector=2, n_matrix=1,
    n_obs_scalar=2, n_obs_vector=1, n_obs_matrix=0,
    vector_size=4, matrix_shape=(3, 3)
)

template_memory = MemoryBank(**memory_cfg.__dict__)
instr_set = InstructionSet([op() for op in ALL_OPS], template_memory)
operators = GeneticOperators(instr_set, rng)

pop_cfg = PopulationConfig(size=20, program_length=(6, 12), elitism=2)
population = Population(pop_cfg, instr_set, memory_cfg, operators=operators, rng=rng)
population.initialize_random()

engine = EvolutionEngine(
    population=population,
    operators=operators,
    evaluator=SymbolicRegressionEvaluator(rng),
    config=EvolutionConfig(max_generations=10, mutation_threshold=0.2)
)
engine.run()
best = population.best_ever
print("Best fitness:", best.fitness)
```

## Module Dependencies

```
operation.py
    ↓
instruction.py ─→ memory_system.py
    ↓                ↓
program.py         individual.py
    ↓                ↓
instruction_set.py → population.py → evolution_engine.py
    ↓                                ↓
operators.py       evaluator.py     agent.py (future)
```

---

## Design Principles

1. **Type Safety**: All operations and instructions are type-safe
2. **Index Wrapping**: Automatic index wrapping prevents errors
3. **Separation of Concerns**: Observation and working registers are separate
4. **Immutability**: Operations are immutable and reusable
5. **Efficiency**: Pre-computed ranges, lazy evaluation, and intron pruning tied to evaluator outputs

---

## Future Development

- Reinforcement-learning specific evaluators (e.g., Flappy Bird integration)
- Environment-facing agent wrappers for inference-time deployment
- Additional diversity metrics and bloat control strategies

---

## Notes

- All documentation is in Markdown format
- Code examples are Python 3.x compatible
- The system uses NumPy for array operations
- All data is stored as `np.float32` for efficiency
- Index wrapping is automatic and transparent

