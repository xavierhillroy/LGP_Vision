# Memory System Documentation

## Overview

The `memory_system.py` module provides the foundational memory management system for Linear Genetic Programming (LGP). It implements a typed memory bank with separate regions for observation (read-only) and working (read/write) registers, as well as a light-weight configuration dataclass for constructing banks on demand.

## Classes

### `MemoryType` (Enum)

An enumeration defining the three types of memory registers supported by the system.

**Values:**
- `SCALAR`: Single floating-point value
- `VECTOR`: 1D array of floating-point values  
- `MATRIX`: 2D array of floating-point values

**Usage:**
```python
from memory_system import MemoryType

print(MemoryType.SCALAR.value)  # "scalar"
print(MemoryType.VECTOR.value)  # "vector"
print(MemoryType.MATRIX.value)  # "matrix"
```

---

### `MemoryConfig`

`MemoryConfig` is a dataclass that captures all size and initialization parameters required to build a `MemoryBank`. It is convenient for factories (e.g. `Individual.random`) that need to spawn fresh memory banks with consistent shapes.

```python
MemoryConfig(
    n_scalar: int,
    n_vector: int,
    n_matrix: int,
    n_obs_scalar: int,
    n_obs_vector: int,
    n_obs_matrix: int,
    vector_size: int,
    matrix_shape: Tuple[int, int],
    init_scalar_range: Tuple[float, float] = (-2.0, 2.0),
    init_vector_range: Tuple[float, float] = (-1.0, 1.0),
    init_matrix_range: Tuple[float, float] = (-0.5, 0.5),
)
```

Call `MemoryBank(**memory_config.__dict__)` to allocate a new bank with those settings.

### `MemoryBank`

The core memory management class that provides typed registers for LGP program execution.

#### Class Responsibilities

1. **Memory Management**: Manages two distinct memory regions:
   - **Observation Registers** (read-only): Store environment inputs and sensor data
   - **Working Registers** (read-write): Store intermediate computation results

2. **Type Safety**: Provides type-specific methods for scalar, vector, and matrix operations

3. **Index Wrapping**: Automatically wraps indices using modulo arithmetic to prevent out-of-bounds errors

4. **Data Handling**: Automatically handles size mismatches when writing vectors/matrices (padding, truncation, center alignment)

5. **State Management**: Provides methods to reset memory and copy memory banks

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_scalar` | `int` | Number of scalar working registers |
| `n_vector` | `int` | Number of vector working registers |
| `n_matrix` | `int` | Number of matrix working registers |
| `n_obs_scalar` | `int` | Number of scalar observation registers |
| `n_obs_vector` | `int` | Number of vector observation registers |
| `n_obs_matrix` | `int` | Number of matrix observation registers |
| `vector_size` | `int` | Size of each vector register |
| `matrix_shape` | `tuple` | Shape (height, width) of each matrix register |
| `scalars` | `np.ndarray` | Working scalar registers (read/write) |
| `vectors` | `np.ndarray` | Working vector registers (read/write) |
| `matrices` | `np.ndarray` | Working matrix registers (read/write) |
| `obs_scalars` | `np.ndarray` | Observation scalar registers (read-only) |
| `obs_vectors` | `np.ndarray` | Observation vector registers (read-only) |
| `obs_matrices` | `np.ndarray` | Observation matrix registers (read-only) |

#### Initialization

```python
MemoryBank(
    n_scalar: int,
    n_vector: int,
    n_matrix: int,
    n_obs_scalar: int,
    n_obs_vector: int,
    n_obs_matrix: int,
    vector_size: int,
    matrix_shape: tuple,
    init_scalar_range: Tuple[float, float] = (-2.0, 2.0),
    init_vector_range: Tuple[float, float] = (-1.0, 1.0),
    init_matrix_range: Tuple[float, float] = (-0.5, 0.5)
)
```

**Parameters:**
- `n_scalar`: Number of scalar working registers
- `n_vector`: Number of vector working registers
- `n_matrix`: Number of matrix working registers
- `n_obs_scalar`: Number of scalar observation registers
- `n_obs_vector`: Number of vector observation registers
- `n_obs_matrix`: Number of matrix observation registers
- `vector_size`: Size of each vector register (all vectors have same size)
- `matrix_shape`: Tuple (height, width) for matrix register dimensions
- `init_scalar_range`: Range for random initialization of scalar working registers
- `init_vector_range`: Range for random initialization of vector working registers
- `init_matrix_range`: Range for random initialization of matrix working registers

**Initialization Behavior:**
- **Observation registers**: Initialized to 0.1 (read-only from program perspective)
- **Working registers**: Initialized with random values in specified ranges (these serve as evolvable constants in LGP)

**Example:**
```python
memory = MemoryBank(
    n_scalar=8,
    n_vector=4,
    n_matrix=2,
    n_obs_scalar=2,
    n_obs_vector=1,
    n_obs_matrix=1,
    vector_size=10,
    matrix_shape=(84, 84)
)
```

---

#### Observation Register Read Methods

These methods read from read-only observation registers (environment inputs).

##### `read_obs_scalar(index: int) -> float`

Read a scalar value from observation registers.

**Parameters:**
- `index`: Register index (wraps around if out of bounds)

**Returns:**
- `float`: The scalar value at the specified index

**Example:**
```python
value = memory.read_obs_scalar(0)
value = memory.read_obs_scalar(10)  # Wraps: 10 % n_obs_scalar
```

##### `read_obs_vector(index: int) -> np.ndarray`

Read a vector from observation registers.

**Parameters:**
- `index`: Register index (wraps around if out of bounds)

**Returns:**
- `np.ndarray`: 1D array of shape `(vector_size,)`

**Example:**
```python
vec = memory.read_obs_vector(0)
print(vec.shape)  # (10,) if vector_size=10
```

##### `read_obs_matrix(index: int) -> np.ndarray`

Read a matrix from observation registers.

**Parameters:**
- `index`: Register index (wraps around if out of bounds)

**Returns:**
- `np.ndarray`: 2D array of shape `matrix_shape`

**Example:**
```python
mat = memory.read_obs_matrix(0)
print(mat.shape)  # (84, 84) if matrix_shape=(84, 84)
```

---

#### Working Register Read Methods

These methods read from read-write working registers.

##### `read_scalar(index: int) -> float`

Read a scalar value from working registers.

**Parameters:**
- `index`: Register index (wraps around if out of bounds)

**Returns:**
- `float`: The scalar value at the specified index

##### `read_vector(index: int) -> np.ndarray`

Read a vector from working registers.

**Parameters:**
- `index`: Register index (wraps around if out of bounds)

**Returns:**
- `np.ndarray`: 1D array of shape `(vector_size,)`

##### `read_matrix(index: int) -> np.ndarray`

Read a matrix from working registers.

**Parameters:**
- `index`: Register index (wraps around if out of bounds)

**Returns:**
- `np.ndarray`: 2D array of shape `matrix_shape`

---

#### Working Register Write Methods

These methods write to working registers. Observation registers cannot be written to by the program.

##### `write_scalar(index: int, value: float) -> None`

Write a scalar value to working registers.

**Parameters:**
- `index`: Register index (wraps around if out of bounds)
- `value`: Scalar value to write

**Example:**
```python
memory.write_scalar(0, 42.0)
memory.write_scalar(100, 3.14)  # Wraps if n_scalar < 100
```

##### `write_vector(index: int, value: np.ndarray) -> None`

Write a vector to working registers.

**Automatic Size Handling:**
- If value is smaller: pads with zeros
- If value is larger: truncates to `vector_size`
- If value matches: copies directly

**Parameters:**
- `index`: Register index (wraps around if out of bounds)
- `value`: Vector to write (can be 1D array, list, or scalar converted to 1D)

**Example:**
```python
memory.write_vector(0, [1, 2, 3])  # Pads if vector_size > 3
memory.write_vector(1, np.arange(20))  # Truncates if vector_size < 20
```

##### `write_matrix(index: int, value: np.ndarray) -> None`

Write a matrix to working registers.

**Automatic Shape Handling (Center Alignment):**
- If value is smaller: pads with zeros (centered)
- If value is larger: crops from center
- If value matches: copies directly

Center alignment preserves spatial relationships in image-like data.

**Parameters:**
- `index`: Register index (wraps around if out of bounds)
- `value`: Matrix to write (will be converted to 2D if needed)

**Example:**
```python
# Write 3x3 matrix to 5x5 register (centered with padding)
small = np.ones((3, 3))
memory.write_matrix(0, small)
# Result: zeros with 3x3 ones in center

# Write 10x10 matrix to 5x5 register (center crop)
large = np.arange(100).reshape(10, 10)
memory.write_matrix(0, large)
# Result: center 5x5 region of original matrix
```

---

#### State Management Methods

##### `load_observation(obs: dict) -> None`

Load observation data into observation registers (read-only memory).

This method updates the observation registers with new environment data. The observation registers are read-only from the program's perspective but can be updated externally via this method.

**Parameters:**
- `obs`: Dictionary containing observation data. Keys can be:
  - `'scalar'`: List or array of scalar values
  - `'vector'`: List or array of vectors (will be converted to 2D)
  - `'matrix'`: List or array of matrices

Only the keys present in the dictionary will be updated. If fewer values are provided than registers, only the first N registers are updated.

**Example:**
```python
memory.load_observation({
    'scalar': [1.5, 2.5],
    'vector': [[1, 2, 3], [4, 5, 6]],
    'matrix': [np.random.rand(84, 84)]
})
```

##### `reset() -> None`

Reset all working registers to initial value (0.1).

Observation registers are left unchanged. This is useful for resetting program state between episodes or iterations while preserving observation data.

**Note:** Registers are reset to 0.1 (not zero) to avoid issues with uninitialized memory in genetic programs.

**Example:**
```python
memory.write_scalar(0, 42.0)
memory.reset()
print(memory.read_scalar(0))  # 0.1
print(memory.read_obs_scalar(0))  # Unchanged
```

##### `reset_all() -> None`

Reset both working and observation registers to initial value (0.1).

This performs a complete reset of all memory, including observations. Use this when starting a completely new episode or simulation.

**Example:**
```python
memory.reset_all()
# All registers (working and observation) are now 0.1
```

##### `copy() -> MemoryBank`

Create a deep copy of this memory bank.

Returns a new `MemoryBank` instance with identical configuration and a complete copy of all register data. The new instance is completely independent - modifications to one will not affect the other.

**Returns:**
- `MemoryBank`: A new MemoryBank instance with copied state

**Example:**
```python
memory2 = memory.copy()
memory2.write_scalar(0, 99.0)
print(memory.read_scalar(0))  # Original unchanged
print(memory2.read_scalar(0))  # 99.0
```

---

#### Special Methods

##### `__repr__() -> str`

Return a string representation of the MemoryBank.

**Format:**
```
MemoryBank(obs=[Xs, Yv, Zm], registers=[As, Bv, Cm])
```
where X/Y/Z are observation counts and A/B/C are working counts.

**Example:**
```
MemoryBank(obs=[2s, 1v, 1m], registers=[8s, 4v, 2m])
```

---

## Key Design Decisions

### 1. Index Wrapping

All index accesses use modulo arithmetic to wrap around if the index exceeds the register count. This prevents `IndexError` exceptions and allows programs to use arbitrary indices, which is useful in genetic programming where programs may be generated with invalid indices.

**Example:**
```python
# If n_scalar = 8
memory.write_scalar(10, 42.0)  # Wraps to index 2 (10 % 8)
value = memory.read_scalar(2)  # 42.0
```

### 2. Observation vs Working Registers

- **Observation registers**: Read-only from program perspective, updated externally via `load_observation()`
- **Working registers**: Read-write, used for program computation

This separation allows the environment to provide inputs without the program overwriting them, while the program can freely manipulate working registers.

### 3. Random Initialization of Working Registers

Working registers are initialized with random values (not zeros) to serve as **evolvable constants** in LGP. These values can be used by the program as constants and may be subject to evolution.

### 4. Automatic Size/Shape Handling

When writing vectors or matrices, the system automatically handles size mismatches:
- **Vectors**: Padding or truncation
- **Matrices**: Center-aligned padding or cropping

This allows operations to work with different-sized inputs without explicit size checking.

### 5. Data Type Consistency

All data is stored as `np.float32` for:
- Memory efficiency
- Consistency with typical LGP implementations
- Compatibility with GPU acceleration if needed

---

## Usage Examples

### Basic Usage

```python
from memory_system import MemoryBank, MemoryType
import numpy as np

# Create memory bank
memory = MemoryBank(
    n_scalar=10,
    n_vector=2,
    n_matrix=1,
    n_obs_scalar=2,
    n_obs_vector=0,
    n_obs_matrix=0,
    vector_size=5,
    matrix_shape=(10, 10)
)

# Load observations
memory.load_observation({
    'scalar': [1.5, 2.5]
})

# Write to working registers
memory.write_scalar(0, 42.0)
memory.write_vector(0, [1, 2, 3, 4, 5])

# Read values
print(memory.read_scalar(0))  # 42.0
print(memory.read_obs_scalar(0))  # 1.5
```

### Working with Programs

```python
# Memory bank is typically used with programs
from program import Program
from instruction import Instruction

# Program execution will read from and write to memory
program.execute(memory)

# After execution, read output from specific register
output = memory.read_scalar(9)  # Assuming output is in register 9
```

### State Management

```python
# Reset working memory between episodes
memory.reset()  # Only resets working registers
# or
memory.reset_all()  # Resets everything

# Copy memory for parallel execution
memory_copy = memory.copy()
```

---

## Notes

- All arrays are stored as `np.float32` for efficiency
- Observation registers are initialized to 0.1 (not zero)
- Working registers are initialized with random values (evolvable constants)
- Index wrapping prevents out-of-bounds errors
- Vector/matrix size mismatches are handled automatically
- Memory banks can be copied for parallel program evaluation

