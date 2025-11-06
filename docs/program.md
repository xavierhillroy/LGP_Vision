# Program Documentation

## Overview

The `program.py` module defines the `Program` class, which represents a complete Linear Genetic Programming (LGP) program as a sequence of instructions. The `Program` class provides execution, analysis, and optimization capabilities including intron (dead code) removal.

## Class: `Program`

### Class Responsibilities

1. **Program Execution**: Execute a sequence of instructions on a memory bank
2. **Intron Detection**: Identify instructions that don't affect the output (dead code)
3. **Intron Removal**: Create optimized programs with introns removed
4. **Dependency Analysis**: Analyze register dependencies between instructions
5. **Program Analysis**: Provide statistics and metrics about program effectiveness

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `instructions` | `List[Instruction]` | List of instructions in the program |
| `_effective_instructions` | `Set[int]` | Cached set of effective instruction indices (lazy) |

---

## Initialization

```python
Program(instructions: List[Instruction])
```

**Parameters:**
- `instructions`: List of `Instruction` objects in execution order

**Example:**
```python
from program import Program
from instruction import Instruction
from operation import ScalarAddOp, ScalarMulOp
from memory_system import MemoryType

# Create instructions
instrs = [
    Instruction(ScalarAddOp(), MemoryType.SCALAR, 0,
               [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),
    Instruction(ScalarMulOp(), MemoryType.SCALAR, 1,
               [MemoryType.SCALAR, MemoryType.SCALAR], [0, 0]),
    Instruction(ScalarAddOp(), MemoryType.SCALAR, 9,
               [MemoryType.SCALAR, MemoryType.SCALAR], [1, 1])
]

# Create program
program = Program(instrs)
```

---

## Core Methods

### `execute(memory: MemoryBank) -> None`

Execute the program on a memory bank.

**Parameters:**
- `memory`: The MemoryBank to execute on

**Behavior:**
- Executes instructions in sequence
- Each instruction reads from and writes to the memory bank
- Execution order is preserved

**Example:**
```python
from memory_system import MemoryBank

memory = MemoryBank(
    n_scalar=10, n_vector=0, n_matrix=0,
    n_obs_scalar=2, n_obs_vector=0, n_obs_matrix=0,
    vector_size=0, matrix_shape=(0, 0)
)

# Load observations
memory.load_observation({'scalar': [2.0, 3.0]})

# Execute program
program.execute(memory)

# Read output from register 9
output = memory.read_scalar(9)
```

---

### `copy() -> Program`

Create a deep copy of the program.

**Returns:**
- `Program`: A new Program instance with copied instructions

**Note:** Operations are immutable and shared between copies.

**Example:**
```python
program_copy = program.copy()
# Modifications to program_copy won't affect program
```

---

## Intron Removal System

### Overview

**Introns** are instructions that don't affect the program's output. They are "dead code" that can be safely removed without changing program behavior.

The intron removal system uses **backward dependency analysis**:
1. Start with output registers
2. Find instructions that write to those registers
3. For each effective instruction, find what it reads
4. Recursively trace back to find all dependencies

### Key Methods

#### `find_effective_instructions(output_registers: List[Tuple[MemoryType, int]]) -> Set[int]`

Find which instructions actually affect the output.

**Parameters:**
- `output_registers`: List of `(MemoryType, index)` tuples specifying output registers

**Returns:**
- `Set[int]`: Set of instruction indices that are effective (non-introns)

**Algorithm:**
- Uses backward dependency analysis
- Tracks which instruction reads each register
- Finds the last writer **before** the instruction that reads it
- Recursively traces dependencies

**Example:**
```python
# Find effective instructions assuming output is in scalar[9]
output_regs = [(MemoryType.SCALAR, 9)]
effective = program.find_effective_instructions(output_regs)
print(effective)  # {0, 1, 2} for example
```

#### `get_introns(output_registers: List[Tuple[MemoryType, int]]) -> Set[int]`

Get instruction indices that are introns (don't affect output).

**Parameters:**
- `output_registers`: List of output register specifications

**Returns:**
- `Set[int]`: Set of instruction indices that are introns

**Example:**
```python
introns = program.get_introns(output_regs)
print(introns)  # {3, 4, 5} for example
```

#### `remove_introns(output_registers: List[Tuple[MemoryType, int]]) -> Program`

Create a new program with introns removed.

**Parameters:**
- `output_registers`: List of output register specifications

**Returns:**
- `Program`: New Program with only effective instructions (maintains order)

**Example:**
```python
compact_program = program.remove_introns(output_regs)
print(len(program))  # 10
print(len(compact_program))  # 5 (if 5 were introns)
```

#### `get_effective_length(output_registers: List[Tuple[MemoryType, int]]) -> int`

Get the number of effective instructions.

**Parameters:**
- `output_registers`: List of output register specifications

**Returns:**
- `int`: Number of effective instructions

**Use Cases:**
- Fitness metrics (smaller effective programs are better)
- Bloat control
- Program analysis

**Example:**
```python
effective_len = program.get_effective_length(output_regs)
print(f"Effective length: {effective_len}/{len(program)}")
```

#### `get_intron_ratio(output_registers: List[Tuple[MemoryType, int]]) -> float`

Get the ratio of intron instructions to total instructions.

**Parameters:**
- `output_registers`: List of output register specifications

**Returns:**
- `float`: Ratio in [0, 1] where 0 = no introns, 1 = all introns

**Example:**
```python
ratio = program.get_intron_ratio(output_regs)
print(f"Intron ratio: {ratio:.1%}")  # e.g., "50.0%"
```

---

## Internal Methods

### `_build_write_map() -> Dict[Tuple[MemoryType, int], List[int]]`

Build a map from registers to instructions that write to them.

**Returns:**
- `Dict`: Maps `(register_type, register_index)` to list of instruction indices

**Purpose:**
- Used for dependency analysis
- Pre-computed once for efficiency

**Example:**
```python
write_map = program._build_write_map()
# { (MemoryType.SCALAR, 0): [0, 2, 5],
#   (MemoryType.SCALAR, 1): [1],
#   ... }
```

### `_get_last_writer(register: Tuple[MemoryType, int], before_idx: int, write_map: Dict) -> int`

Find the last instruction that wrote to a register before a given index.

**Parameters:**
- `register`: `(MemoryType, index)` tuple
- `before_idx`: Only consider instructions before this index
- `write_map`: Pre-computed write map

**Returns:**
- `int`: Instruction index, or -1 if no writer found

**Purpose:**
- Used in dependency analysis to find which instruction produced a value
- Critical for handling overwritten registers correctly

---

## Pretty Printing

### `to_string(output_registers: List[Tuple[MemoryType, int]] = None, show_introns: bool = True) -> str`

Pretty print the program, optionally marking introns.

**Parameters:**
- `output_registers`: If provided, compute and show introns
- `show_introns`: If True, mark intron instructions with 'X'

**Returns:**
- `str`: Formatted string representation

**Format:**
```
  0: instruction1
  1: instruction2
X 2: instruction3  (if intron)
  3: instruction4

Effective: 3/4 (75.0%)
```

**Example:**
```python
output = program.to_string(output_regs, show_introns=True)
print(output)
```

---

## Special Methods

### `__len__() -> int`

Return the number of instructions in the program.

**Example:**
```python
print(len(program))  # 10
```

### `__repr__() -> str`

Return a string representation of the program.

**Format:**
```
Program(N instructions)
```

**Example:**
```python
print(program)  # Program(10 instructions)
```

---

## Usage Examples

### Basic Program Execution

```python
from program import Program
from instruction import Instruction
from operation import ScalarAddOp, ScalarMulOp
from memory_system import MemoryBank, MemoryType

# Create program
instructions = [
    Instruction(ScalarAddOp(), MemoryType.SCALAR, 0,
               [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),
    Instruction(ScalarMulOp(), MemoryType.SCALAR, 1,
               [MemoryType.SCALAR, MemoryType.SCALAR], [0, 0]),
    Instruction(ScalarAddOp(), MemoryType.SCALAR, 9,
               [MemoryType.SCALAR, MemoryType.SCALAR], [1, 1])
]

program = Program(instructions)

# Execute
memory = MemoryBank(...)
memory.load_observation({'scalar': [2.0, 3.0]})
program.execute(memory)

# Read output
output = memory.read_scalar(9)
```

### Intron Removal

```python
# Create program with introns
instructions = [
    # Effective chain
    Instruction(ScalarAddOp(), MemoryType.SCALAR, 0,
               [MemoryType.SCALAR, MemoryType.SCALAR], [-1, -2]),
    Instruction(ScalarMulOp(), MemoryType.SCALAR, 1,
               [MemoryType.SCALAR, MemoryType.SCALAR], [0, 0]),
    Instruction(ScalarAddOp(), MemoryType.SCALAR, 9,
               [MemoryType.SCALAR, MemoryType.SCALAR], [1, 1]),
    # Introns (don't affect scalar[9])
    Instruction(ScalarSubOp(), MemoryType.SCALAR, 2,
               [MemoryType.SCALAR, MemoryType.SCALAR], [0, 1]),
    Instruction(ScalarMulOp(), MemoryType.SCALAR, 3,
               [MemoryType.SCALAR, MemoryType.SCALAR], [2, 2])
]

program = Program(instructions)

# Find introns
output_regs = [(MemoryType.SCALAR, 9)]
effective = program.find_effective_instructions(output_regs)
print(f"Effective: {effective}")  # {0, 1, 2}

introns = program.get_introns(output_regs)
print(f"Introns: {introns}")  # {3, 4}

# Remove introns
compact = program.remove_introns(output_regs)
print(len(compact))  # 3

# Get statistics
print(f"Effective length: {program.get_effective_length(output_regs)}")
print(f"Intron ratio: {program.get_intron_ratio(output_regs):.1%}")
```

### Program Analysis

```python
# Analyze program
output_regs = [(MemoryType.SCALAR, 9)]

# Get statistics
effective_len = program.get_effective_length(output_regs)
intron_ratio = program.get_intron_ratio(output_regs)

print(f"Total instructions: {len(program)}")
print(f"Effective instructions: {effective_len}")
print(f"Intron ratio: {intron_ratio:.1%}")

# Pretty print with intron marking
print(program.to_string(output_regs, show_introns=True))
```

---

## Design Notes

### Intron Removal Algorithm

The intron removal uses **backward dependency analysis**:

1. **Write Map**: Pre-computes which instructions write to each register
2. **Backward Tracing**: Starts from output registers and traces backwards
3. **Reader Context**: Tracks which instruction reads each register to find the correct writer
4. **Worklist Algorithm**: Uses a worklist to process all dependencies

**Key Insight**: When tracing a register, we find the last writer **before** the instruction that reads it, not just the last writer before the end of the program. This correctly handles cases where registers are overwritten.

**Example:**
```
Instruction 0: scalar[0] = obs[-1] + obs[-2]  (effective)
Instruction 1: scalar[1] = scalar[0] * scalar[0]  (reads [0] at pos 1, effective)
Instruction 2: scalar[0] = scalar[1] - scalar[1]  (overwrites [0] at pos 2)
Instruction 3: scalar[9] = scalar[1] + scalar[1]  (output, effective)
```

When tracing `scalar[1]` (read by instruction 3), we find instruction 1 writes it.
When tracing `scalar[0]` (read by instruction 1), we find instruction 0 writes it (before instruction 1), not instruction 2.

### Multiple Output Registers

The system supports multiple output registers. All dependencies leading to any output register are traced, making all relevant instructions effective.

### Overwritten Registers

The algorithm correctly handles registers that are overwritten multiple times by tracking which instruction reads each register and finding the writer before that specific reader.

---

## Notes

- Instructions are executed in order
- Intron removal maintains instruction order in the output
- The `_effective_instructions` cache is lazy (computed on first use)
- Operations are immutable and shared between program copies
- All dependency analysis is done statically (no execution required)

