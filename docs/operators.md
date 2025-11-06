# Genetic Operators Documentation

## Overview

The `operators.py` module defines genetic operators for evolving Linear Genetic Programming (LGP) programs. Currently, this module is a placeholder with the `GeneticOperators` class structure defined but not fully implemented.

## Class: `GeneticOperators`

### Class Responsibilities (Planned)

1. **Mutation**: Mutate instructions in programs
2. **Crossover**: Combine two programs to create offspring
3. **Selection**: Select individuals from population
4. **Program Variation**: Generate variations of existing programs

### Current Status

**Status:** Placeholder/Stub Implementation

The class is currently a skeleton with minimal implementation. The `micro_mutate` method is defined but not implemented.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `instruction_set` | `InstructionSet` | Instruction set for generating new instructions |

---

## Initialization

```python
GeneticOperators(instruction_set: InstructionSet)
```

**Parameters:**
- `instruction_set`: InstructionSet instance for generating new instructions

**Example:**
```python
from operators import GeneticOperators
from instruction_set import InstructionSet
from memory_system import MemoryBank
from operation import ALL_OPS

# Create instruction set
memory = MemoryBank(...)
operations = [op() for op in ALL_OPS]
instruct_set = InstructionSet(operations, memory)

# Create genetic operators
genetic_ops = GeneticOperators(instruct_set)
```

---

## Planned Methods

### `micro_mutate(instruction: Instruction, rate: float) -> Instruction`

**Status:** Stub (not implemented)

**Intended Purpose:**
- Mutate a single instruction with a given mutation rate
- Possible mutations:
  - Change operation
  - Change destination register
  - Change source registers
  - Change operation parameters

**Parameters:**
- `instruction`: The instruction to mutate
- `rate`: Mutation rate (probability of mutation)

**Returns:**
- `Instruction`: Mutated instruction (or original if not mutated)

---

## Future Implementation Notes

### Mutation Strategies

Potential mutation strategies to implement:

1. **Operation Mutation**: Replace operation with random compatible operation
2. **Register Mutation**: Change source or destination registers
3. **Instruction Replacement**: Replace entire instruction with random one
4. **Parameter Mutation**: Modify operation parameters (if applicable)

### Crossover Strategies

Potential crossover strategies:

1. **One-Point Crossover**: Split programs at random point, swap segments
2. **Two-Point Crossover**: Swap middle segment between two programs
3. **Uniform Crossover**: Randomly select instructions from each parent
4. **Instruction-Level Crossover**: Swap individual instructions

### Selection Strategies

Potential selection strategies:

1. **Tournament Selection**: Random tournament, best individual wins
2. **Roulette Wheel Selection**: Probability proportional to fitness
3. **Rank Selection**: Probability based on rank
4. **Elitism**: Always keep best individuals

---

## Notes

- This module is currently a placeholder
- Implementation is planned for future development
- Genetic operators are essential for evolutionary algorithms
- The class structure is defined but methods need implementation

