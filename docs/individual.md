# Individual Documentation

## Overview

`Individual` is the container for candidate solutions in the system. Each individual bundles:

- A `Program` (sequence of instructions)
- A `MemoryBank` holding evolvable constants and observation registers
- Metadata used by the evolutionary loop (id, fitness, age, lineage)

## Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `program` | `Program` | Executable LGP program |
| `memory` | `MemoryBank` | Working/evolutionary registers |
| `fitness` | `Optional[float]` | Cached fitness value |
| `age` | `int` | Generational age |
| `parent_ids` | `Tuple[int, ...]` | Lineage tracking |

## Factory Method

`Individual.random(...)` constructs fresh candidates:

```python
@classmethod
def random(
    cls,
    instruction_set: InstructionSet,
    memory_config: MemoryConfig,
    program_length: int,
    rng: Optional[np.random.Generator] = None,
    mutate_constants: bool = True,
) -> Individual:
    ...
```

Steps:
1. Draw a random program from the supplied `InstructionSet`.
2. Allocate a new `MemoryBank` using the provided `MemoryConfig`.
3. Optionally call `GeneticOperators.mutate_constants` for a stochastic constant shake-up.

This keeps population initialization concise and makes it easy to spawn reproducible individuals by passing a seeded RNG.

## Evaluation Helpers

- `evaluate(evaluator)` defers to any `FitnessEvaluator` implementation.
- `invalidate_fitness()` clears the cached score when the individual mutates.

## Copying and Offspring

- `copy(new_id: bool = True)` deep-copies both program and memory.
- `create_offspring(parent_ids)` is a convenience wrapper used by the evolution engine to reset lineage, age, and fitness.

## Usage Example

```python
from Individual import Individual
from instruction_set import InstructionSet
from memory_system import MemoryConfig

cfg = MemoryConfig(...)
instr_set = InstructionSet(operations, template_memory)
ind = Individual.random(instr_set, cfg, program_length=8, rng=rng)
print(ind.program.to_string())
```
