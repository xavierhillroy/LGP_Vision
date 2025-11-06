# Instruction Set Documentation

## Overview

The `instruction_set.py` module provides the `InstructionSet` class, which is responsible for generating random, type-safe instructions for Linear Genetic Programming (LGP). It ensures that generated instructions respect the memory bank configuration and the distinction between observation (read-only) and working (read-write) registers.

## Class: `InstructionSet`

### Class Responsibilities

1. **Random Instruction Generation**: Generate random, type-safe instructions
2. **Program Generation**: Generate random programs of specified length
3. **Memory Configuration Management**: Track memory bank configuration for valid register selection
4. **Type Safety Enforcement**: Ensure generated instructions are type-safe by construction

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `operations` | `List[Operation]` | List of operation instances to use |
| `n_scalar` | `int` | Number of scalar working registers |
| `n_vector` | `int` | Number of vector working registers |
| `n_matrix` | `int` | Number of matrix working registers |
| `n_obs_scalar` | `int` | Number of scalar observation registers |
| `n_obs_vector` | `int` | Number of vector observation registers |
| `n_obs_matrix` | `int` | Number of matrix observation registers |
| `_source_ranges` | `Dict[MemoryType, List[int]]` | Pre-computed valid source indices for each type |
| `_dest_ranges` | `Dict[MemoryType, List[int]]` | Pre-computed valid destination indices for each type |

---

## Initialization

```python
InstructionSet(
    operations: List[Operation],
    memory: MemoryBank
)
```

**Parameters:**
- `operations`: List of operation instances to use for instruction generation
- `memory`: MemoryBank instance to read configuration from

**Behavior:**
- Reads memory bank configuration (register counts)
- Pre-computes valid index ranges for source and destination registers
- Source ranges include both observation (negative) and working (non-negative) indices
- Destination ranges only include working register indices (non-negative)

**Example:**
```python
from instruction_set import InstructionSet
from memory_system import MemoryBank
from operation import ALL_OPS

# Create memory bank
memory = MemoryBank(
    n_scalar=10, n_vector=4, n_matrix=2,
    n_obs_scalar=2, n_obs_vector=1, n_obs_matrix=1,
    vector_size=10, matrix_shape=(84, 84)
)

# Create operations list
operations = [op() for op in ALL_OPS]

# Create instruction set
instruct_set = InstructionSet(operations, memory)
```

---

## Methods

### `generate_random_instruction(rng=None) -> Instruction`

Generate a random type-safe instruction.

**Parameters:**
- `rng`: Optional numpy random generator (creates new one if None)

**Returns:**
- `Instruction`: A randomly generated, type-safe instruction

**Algorithm:**
1. Randomly selects an operation from the available operations
2. For each input type required by the operation, randomly selects a valid source register
3. Randomly selects a valid destination register matching the operation's output type
4. Creates and returns the instruction

**Source Register Selection:**
- Can read from both observation and working registers
- Observation registers: negative indices (e.g., -1, -2, ...)
- Working registers: non-negative indices (e.g., 0, 1, 2, ...)

**Destination Register Selection:**
- Can only write to working registers (non-negative indices)
- Observation registers are read-only

**Example:**
```python
import numpy as np

# Generate random instruction
rng = np.random.default_rng(42)
instr = instruct_set.generate_random_instruction(rng)
print(instr)
# Output: scalar[5] = scalar_add(obs_scalar[-1], scalar[2])
```

---

### `generate_random_program(length: int, rng=None) -> Program`

Generate a random program of specified length.

**Parameters:**
- `length`: Number of instructions in the program
- `rng`: Optional numpy random generator (creates new one if None)

**Returns:**
- `Program`: A randomly generated program

**Example:**
```python
import numpy as np

# Generate random program with 10 instructions
rng = np.random.default_rng(42)
program = instruct_set.generate_random_program(10, rng)
print(f"Program length: {len(program)}")
print(program)
```

---

### `get_random_operator(rng=None) -> Operation`

Get a random operation from the available operations.

**Parameters:**
- `rng`: Optional numpy random generator (creates new one if None)

**Returns:**
- `Operation`: A randomly selected operation instance

**Example:**
```python
import numpy as np

rng = np.random.default_rng()
op = instruct_set.get_random_operator(rng)
print(op.name)
```

---

### `get_random_dest(dest_type: MemoryType, rng=None) -> int`

Get a random valid destination register index for a given type.

**Parameters:**
- `dest_type`: The memory type (SCALAR, VECTOR, or MATRIX)
- `rng`: Optional numpy random generator (creates new one if None)

**Returns:**
- `int`: A random valid destination register index (non-negative)

**Note:** Only returns working register indices (observation registers are read-only).

**Example:**
```python
import numpy as np
from memory_system import MemoryType

rng = np.random.default_rng()
dest_idx = instruct_set.get_random_dest(MemoryType.SCALAR, rng)
print(f"Random destination: {dest_idx}")  # e.g., 5
```

---

### `get_random_source(source_type: MemoryType, rng=None) -> int`

Get a random valid source register index for a given type.

**Parameters:**
- `source_type`: The memory type (SCALAR, VECTOR, or MATRIX)
- `rng`: Optional numpy random generator (creates new one if None)

**Returns:**
- `int`: A random valid source register index (can be negative for observations)

**Note:** Returns both observation (negative) and working (non-negative) register indices.

**Example:**
```python
import numpy as np
from memory_system import MemoryType

rng = np.random.default_rng()
src_idx = instruct_set.get_random_source(MemoryType.SCALAR, rng)
print(f"Random source: {src_idx}")  # e.g., -1 (observation) or 5 (working)
```

---

## Internal Data Structures

### Source Ranges

Pre-computed valid source register indices for each memory type:

```python
_source_ranges = {
    MemoryType.SCALAR: [-n_obs_scalar, ..., -1, 0, ..., n_scalar-1],
    MemoryType.VECTOR: [-n_obs_vector, ..., -1, 0, ..., n_vector-1],
    MemoryType.MATRIX: [-n_obs_matrix, ..., -1, 0, ..., n_matrix-1]
}
```

**Format:**
- Negative indices: Observation registers (read-only)
- Non-negative indices: Working registers (read-write)

### Destination Ranges

Pre-computed valid destination register indices for each memory type:

```python
_dest_ranges = {
    MemoryType.SCALAR: [0, 1, ..., n_scalar-1],
    MemoryType.VECTOR: [0, 1, ..., n_vector-1],
    MemoryType.MATRIX: [0, 1, ..., n_matrix-1]
}
```

**Format:**
- Only non-negative indices (working registers only)
- Observation registers are read-only and cannot be written to

---

## Usage Examples

### Basic Usage

```python
from instruction_set import InstructionSet
from memory_system import MemoryBank
from operation import ALL_OPS
import numpy as np

# Setup
memory = MemoryBank(
    n_scalar=10, n_vector=4, n_matrix=2,
    n_obs_scalar=2, n_obs_vector=1, n_obs_matrix=1,
    vector_size=10, matrix_shape=(84, 84)
)

operations = [op() for op in ALL_OPS]
instruct_set = InstructionSet(operations, memory)

# Generate random instruction
rng = np.random.default_rng(42)
instr = instruct_set.generate_random_instruction(rng)
print(instr)
```

### Generating Programs

```python
# Generate random program
rng = np.random.default_rng(42)
program = instruct_set.generate_random_program(20, rng)

print(f"Generated program with {len(program)} instructions")
print(program.to_string())
```

### Custom Random Generation

```python
# Use specific random generator for reproducibility
rng = np.random.default_rng(seed=42)

# Generate multiple instructions with same RNG
instr1 = instruct_set.generate_random_instruction(rng)
instr2 = instruct_set.generate_random_instruction(rng)
instr3 = instruct_set.generate_random_instruction(rng)

# Generate program
program = instruct_set.generate_random_program(10, rng)
```

### Using Helper Methods

```python
# Get random operator
op = instruct_set.get_random_operator(rng)
print(f"Random operation: {op.name}")

# Get random destination
from memory_system import MemoryType
dest_idx = instruct_set.get_random_dest(MemoryType.SCALAR, rng)
print(f"Random scalar destination: {dest_idx}")

# Get random source
src_idx = instruct_set.get_random_source(MemoryType.VECTOR, rng)
print(f"Random vector source: {src_idx}")
```

---

## Design Notes

### Type Safety

All generated instructions are **type-safe by construction**:
- Operation input types are matched with source register types
- Operation output types are matched with destination register types
- Invalid combinations are impossible

### Memory Configuration

The instruction set reads memory configuration from the `MemoryBank` instance:
- Ensures generated instructions use valid register indices
- Pre-computes valid ranges for efficiency
- Automatically handles observation vs working register distinction

### Observation vs Working Registers

The instruction set enforces the distinction:
- **Source registers**: Can be observation (negative) or working (non-negative)
- **Destination registers**: Can only be working (non-negative)

This ensures programs cannot write to observation registers while allowing flexible reading.

### Pre-computed Ranges

Valid index ranges are pre-computed for efficiency:
- Avoids repeated computation during instruction generation
- Enables fast random selection
- Maintains consistency with memory bank configuration

---

## Notes

- All generated instructions are type-safe
- Instructions can read from both observation and working registers
- Instructions can only write to working registers
- Random generation uses numpy's random number generator
- Pre-computed ranges ensure efficient generation
- Memory configuration is read from the MemoryBank instance

