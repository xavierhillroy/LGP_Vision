# Instruction Documentation

## Overview

The `instruction.py` module defines the `Instruction` class, which represents a single instruction in a Linear Genetic Programming (LGP) program. An instruction specifies an operation to perform, where to read inputs from, and where to write the output.

## Class: `Instruction`

### Class Responsibilities

1. **Operation Execution**: Execute an operation with inputs from memory registers
2. **Type Safety**: Ensure instruction is type-safe (operation types match register types)
3. **Memory Access**: Handle both observation (read-only) and working (read-write) registers
4. **Index Wrapping**: Automatically wrap indices to prevent out-of-bounds errors
5. **Dependency Analysis**: Provide methods to analyze read/write dependencies

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `operation` | `Operation` | The operation to execute |
| `dest_type` | `MemoryType` | Type of destination register |
| `dest_index` | `int` | Index of destination register (must be ≥ 0) |
| `source_types` | `List[MemoryType]` | Types of source registers |
| `source_indices` | `List[int]` | Indices of source registers (negative = observation) |

### Register Indexing

The instruction system uses a special indexing scheme:

- **Negative indices** (e.g., -1, -2, ...): Observation registers (read-only)
- **Non-negative indices** (e.g., 0, 1, 2, ...): Working registers (read-write)

This allows instructions to read from both observation and working registers, but only write to working registers.

---

## Initialization

```python
Instruction(
    operation: Operation,
    dest_type: MemoryType,
    dest_index: int,
    source_types: List[MemoryType],
    source_indices: List[int]
)
```

**Parameters:**
- `operation`: The operation instance to execute
- `dest_type`: Type of the destination register (SCALAR, VECTOR, or MATRIX)
- `dest_index`: Index of destination register (must be non-negative)
- `source_types`: List of source register types (must match operation's input types)
- `source_indices`: List of source register indices (negative for observations)

**Validation:**
- `dest_index` must be ≥ 0 (cannot write to observation registers)
- `len(source_types)` must equal `len(source_indices)`
- Type safety is checked by `is_valid()` method

**Example:**
```python
from instruction import Instruction
from operation import ScalarAddOp
from memory_system import MemoryType

# Create instruction: scalar[5] = scalar_add(obs_scalar[-1], scalar[0])
instr = Instruction(
    operation=ScalarAddOp(),
    dest_type=MemoryType.SCALAR,
    dest_index=5,
    source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
    source_indices=[-1, 0]  # -1 = observation, 0 = working register
)
```

---

## Methods

### `execute(memory: MemoryBank) -> None`

Execute the instruction on a memory bank.

**Parameters:**
- `memory`: The MemoryBank to read from and write to

**Behavior:**
1. Reads source operands from memory (handling observation vs working registers)
2. Executes the operation with the operands
3. Writes the result to the destination register

**Index Wrapping:**
- All indices are automatically wrapped using modulo arithmetic
- Prevents `IndexError` exceptions for out-of-bounds indices

**Example:**
```python
from memory_system import MemoryBank
from instruction import Instruction
from operation import ScalarMulOp

# Create memory and instruction
memory = MemoryBank(
    n_scalar=10, n_vector=0, n_matrix=0,
    n_obs_scalar=2, n_obs_vector=0, n_obs_matrix=0,
    vector_size=0, matrix_shape=(0, 0)
)

# Load observations
memory.load_observation({'scalar': [2.0, 3.0]})

# Create instruction: scalar[0] = scalar_mul(obs_scalar[-1], obs_scalar[-2])
instr = Instruction(
    operation=ScalarMulOp(),
    dest_type=MemoryType.SCALAR,
    dest_index=0,
    source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
    source_indices=[-1, -2]  # Read from observations
)

# Execute instruction
instr.execute(memory)

# Result is in scalar[0]
print(memory.read_scalar(0))  # 6.0 (2.0 * 3.0)
```

---

### `is_valid() -> bool`

Check if the instruction is type-safe.

**Returns:**
- `bool`: `True` if instruction is valid, `False` otherwise

**Validation Checks:**
1. Destination index is non-negative (cannot write to observations)
2. Operation's output type matches destination type
3. Operation's input types match source types

**Example:**
```python
# Valid instruction
instr1 = Instruction(ScalarAddOp(), MemoryType.SCALAR, 0,
                    [MemoryType.SCALAR, MemoryType.SCALAR], [0, 1])
print(instr1.is_valid())  # True

# Invalid: output type mismatch
instr2 = Instruction(ScalarAddOp(), MemoryType.VECTOR, 0,
                    [MemoryType.SCALAR, MemoryType.SCALAR], [0, 1])
print(instr2.is_valid())  # False
```

---

### `uses_observation() -> bool`

Check if this instruction reads from any observation register.

**Returns:**
- `bool`: `True` if any source index is negative (observation register)

**Example:**
```python
# Uses observation
instr1 = Instruction(ScalarAddOp(), MemoryType.SCALAR, 0,
                    [MemoryType.SCALAR, MemoryType.SCALAR], [-1, 0])
print(instr1.uses_observation())  # True

# Doesn't use observation
instr2 = Instruction(ScalarAddOp(), MemoryType.SCALAR, 0,
                    [MemoryType.SCALAR, MemoryType.SCALAR], [0, 1])
print(instr2.uses_observation())  # False
```

---

### `get_read_registers() -> List[tuple]`

Get all registers this instruction reads from.

**Returns:**
- `List[tuple]`: List of `(MemoryType, index)` tuples for each source register

**Example:**
```python
instr = Instruction(ScalarAddOp(), MemoryType.SCALAR, 0,
                   [MemoryType.SCALAR, MemoryType.VECTOR], [-1, 2])
reads = instr.get_read_registers()
# Returns: [(MemoryType.SCALAR, -1), (MemoryType.VECTOR, 2)]
```

---

### `get_write_register() -> tuple`

Get the register this instruction writes to.

**Returns:**
- `tuple`: `(MemoryType, index)` tuple for the destination register

**Example:**
```python
instr = Instruction(ScalarAddOp(), MemoryType.SCALAR, 5,
                   [MemoryType.SCALAR, MemoryType.SCALAR], [0, 1])
writes = instr.get_write_register()
# Returns: (MemoryType.SCALAR, 5)
```

---

## String Representations

### `__repr__() -> str`

Human-readable representation of the instruction.

**Format:**
```
dest_type[dest_index] = operation_name(source_operands)
```

**Example:**
```python
instr = Instruction(ScalarAddOp(), MemoryType.SCALAR, 5,
                   [MemoryType.SCALAR, MemoryType.SCALAR], [-1, 0])
print(instr)
# Output: scalar[5] = scalar_add(obs_scalar[-1], scalar[0])
```

### `to_compact_str() -> str`

Compact string representation for logging/debugging.

**Format:**
```
dest=operation_name(source_list)
```

**Abbreviations:**
- `s` = scalar
- `v` = vector
- `m` = matrix
- `o` prefix = observation register

**Example:**
```python
instr = Instruction(ScalarAddOp(), MemoryType.SCALAR, 5,
                   [MemoryType.SCALAR, MemoryType.SCALAR], [-1, 0])
print(instr.to_compact_str())
# Output: s5=scalar_add(os1,s0)
```

---

## Index Wrapping Behavior

All register indices are automatically wrapped using modulo arithmetic:

### Observation Registers

- **Scalar**: `obs_idx % n_obs_scalar`
- **Vector**: `obs_idx % n_obs_vector`
- **Matrix**: `obs_idx % n_obs_matrix`

### Working Registers

- **Scalar**: `index % n_scalar`
- **Vector**: `index % n_vector`
- **Matrix**: `index % n_matrix`

**Example:**
```python
# If n_scalar = 8
instr = Instruction(ScalarAddOp(), MemoryType.SCALAR, 10,  # Wraps to index 2
                   [MemoryType.SCALAR, MemoryType.SCALAR], [0, 1])
# dest_index 10 wraps to 10 % 8 = 2
```

This prevents `IndexError` exceptions and allows programs with arbitrary indices.

---

## Memory Access Patterns

### Reading from Registers

The instruction reads source operands based on index sign:

```python
for src_type, src_idx in zip(source_types, source_indices):
    if src_idx < 0:
        # Observation register (read-only)
        obs_idx = abs(src_idx) - 1
        # Read from obs_scalars, obs_vectors, or obs_matrices
    else:
        # Working register (read-write)
        # Read from scalars, vectors, or matrices
```

### Writing to Registers

The instruction always writes to working registers:

```python
# dest_index must be >= 0
if dest_type == MemoryType.SCALAR:
    memory.write_scalar(dest_index, result)
elif dest_type == MemoryType.VECTOR:
    memory.write_vector(dest_index, result)
elif dest_type == MemoryType.MATRIX:
    memory.write_matrix(dest_index, result)
```

---

## Usage Examples

### Basic Instruction Execution

```python
from instruction import Instruction
from operation import ScalarAddOp, ScalarMulOp
from memory_system import MemoryBank, MemoryType

# Create memory
memory = MemoryBank(
    n_scalar=10, n_vector=0, n_matrix=0,
    n_obs_scalar=2, n_obs_vector=0, n_obs_matrix=0,
    vector_size=0, matrix_shape=(0, 0)
)

# Load observations
memory.load_observation({'scalar': [5.0, 3.0]})

# Instruction 1: Read from observations, write to working register
instr1 = Instruction(
    operation=ScalarAddOp(),
    dest_type=MemoryType.SCALAR,
    dest_index=0,
    source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
    source_indices=[-1, -2]  # obs[-1] = 5.0, obs[-2] = 3.0
)
instr1.execute(memory)
print(memory.read_scalar(0))  # 8.0

# Instruction 2: Read from working register, write to another
instr2 = Instruction(
    operation=ScalarMulOp(),
    dest_type=MemoryType.SCALAR,
    dest_index=1,
    source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
    source_indices=[0, 0]  # scalar[0] * scalar[0]
)
instr2.execute(memory)
print(memory.read_scalar(1))  # 64.0 (8.0 * 8.0)
```

### Type Safety Checking

```python
from instruction import Instruction
from operation import ScalarAddOp, VectorAddOp
from memory_system import MemoryType

# Valid: scalar operation -> scalar output
instr1 = Instruction(
    operation=ScalarAddOp(),
    dest_type=MemoryType.SCALAR,
    dest_index=0,
    source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
    source_indices=[0, 1]
)
print(instr1.is_valid())  # True

# Invalid: scalar operation -> vector output (type mismatch)
instr2 = Instruction(
    operation=ScalarAddOp(),
    dest_type=MemoryType.VECTOR,  # Wrong type!
    dest_index=0,
    source_types=[MemoryType.SCALAR, MemoryType.SCALAR],
    source_indices=[0, 1]
)
print(instr2.is_valid())  # False
```

### Dependency Analysis

```python
from instruction import Instruction
from operation import ScalarAddOp

# Create instruction
instr = Instruction(
    operation=ScalarAddOp(),
    dest_type=MemoryType.SCALAR,
    dest_index=5,
    source_types=[MemoryType.SCALAR, MemoryType.VECTOR],
    source_indices=[-1, 2]
)

# Analyze dependencies
reads = instr.get_read_registers()
# [(MemoryType.SCALAR, -1), (MemoryType.VECTOR, 2)]

writes = instr.get_write_register()
# (MemoryType.SCALAR, 5)

uses_obs = instr.uses_observation()
# True (reads from obs_scalar[-1])
```

---

## Design Notes

### Separation of Observation and Working Registers

The negative index convention separates observation (read-only) and working (read-write) registers:
- **Negative indices**: Observation registers (environment inputs)
- **Non-negative indices**: Working registers (program computation)

This ensures programs cannot overwrite observation data while allowing flexible access to both register types.

### Type Safety

Instructions are validated for type safety:
- Operation input types must match source register types
- Operation output type must match destination register type
- Type mismatches are caught early (via `is_valid()`)

### Index Wrapping

All indices are wrapped to prevent out-of-bounds errors. This is essential in genetic programming where programs may be generated with arbitrary indices.

### Operation Immutability

Operations are stateless and immutable. A single operation instance can be shared across multiple instructions, which is memory-efficient.

---

## Notes

- Instructions are dataclasses (Python `@dataclass`)
- Source indices are automatically converted: `obs_idx = abs(src_idx) - 1` for observations
- All memory access uses modulo wrapping for safety
- Instructions can be created without execution (for program construction)
- Type safety is enforced but not checked at construction time (use `is_valid()`)

