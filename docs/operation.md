# Operation Documentation

## Overview

The `operation.py` module defines the operation system for Linear Genetic Programming (LGP). It provides an abstract base class for operations and implements a comprehensive set of typed operations that work with scalars, vectors, and matrices.

## Architecture

### Abstract Base Class

All operations inherit from the `Operation` abstract base class, which enforces a consistent interface for:
- Type information (input/output types)
- Execution logic
- Metadata (name, differentiability)

### Type System

Operations are type-safe: each operation specifies:
- **Input types**: List of `MemoryType` values (SCALAR, VECTOR, MATRIX)
- **Output type**: Single `MemoryType` value

This ensures that instructions can only be created with compatible operations.

---

## Abstract Base Class: `Operation`

### Class Responsibilities

1. **Type Specification**: Define input and output types
2. **Execution**: Perform the actual computation
3. **Metadata**: Provide name and differentiability information

### Abstract Methods

All subclasses must implement:

#### `input_types() -> List[MemoryType]`

Returns the list of input types required by this operation.

**Returns:**
- `List[MemoryType]`: List of memory types for each input operand

**Example:**
```python
class ScalarAddOp(Operation):
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
```

#### `output_type() -> MemoryType`

Returns the output type produced by this operation.

**Returns:**
- `MemoryType`: The memory type of the output

**Example:**
```python
class ScalarAddOp(Operation):
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
```

#### `execute(*inputs) -> Any`

Executes the operation with the given inputs.

**Parameters:**
- `*inputs`: Variable number of inputs matching the operation's input types

**Returns:**
- Result of the operation (type matches `output_type()`)

**Example:**
```python
class ScalarAddOp(Operation):
    def execute(self, a: float, b: float) -> float:
        return a + b
```

#### `name: str` (property)

Returns the name of the operation.

**Returns:**
- `str`: Operation name (e.g., "scalar_add")

#### `differentiable: bool` (property)

Returns whether the operation is differentiable.

**Returns:**
- `bool`: `True` if differentiable, `False` otherwise

**Note:** Operations that are not differentiable at certain points (e.g., `max`, `abs` at zero) return `False`.

### Special Methods

#### `__repr__() -> str`

Returns a string representation of the operation.

**Format:**
```
OperationName(input_types) -> output_type
```

**Example:**
```
scalar_add(scalar, scalar) -> scalar
```

---

## Operation Categories

Operations are organized into four categories:

1. **Scalar Operations**: Operate on scalar values
2. **Vector Operations**: Operate on vectors (1D arrays)
3. **Matrix Operations**: Operate on matrices (2D arrays)
4. **Cross-Type Operations**: Operate across different types (e.g., scalar × vector)

---

## Scalar Operations

All scalar operations work with `float` values.

### Arithmetic Operations

#### `ScalarAddOp`

**Description:** Add two scalar values

**Signature:** `(scalar, scalar) -> scalar`

**Formula:** `a + b`

**Differentiable:** Yes

**Example:**
```python
op = ScalarAddOp()
result = op.execute(3.0, 2.0)  # 5.0
```

#### `ScalarSubOp`

**Description:** Subtract two scalar values

**Signature:** `(scalar, scalar) -> scalar`

**Formula:** `a - b`

**Differentiable:** Yes

**Example:**
```python
op = ScalarSubOp()
result = op.execute(5.0, 3.0)  # 2.0
```

#### `ScalarMulOp`

**Description:** Multiply two scalars

**Signature:** `(scalar, scalar) -> scalar`

**Formula:** `a * b`

**Differentiable:** Yes

**Example:**
```python
op = ScalarMulOp()
result = op.execute(3.0, 4.0)  # 12.0
```

#### `ScalarDivProtectedOp`

**Description:** Protected division (returns 1.0 if divisor is zero)

**Signature:** `(scalar, scalar) -> scalar`

**Formula:** `a / b if |b| > 1e-8 else 1.0`

**Differentiable:** No (discontinuity at b=0)

**Example:**
```python
op = ScalarDivProtectedOp()
result1 = op.execute(10.0, 2.0)  # 5.0
result2 = op.execute(10.0, 0.0)  # 1.0 (protected)
```

### Comparison Operations

#### `ScalarMaxOp`

**Description:** Maximum of two scalars

**Signature:** `(scalar, scalar) -> scalar`

**Formula:** `max(a, b)`

**Differentiable:** No (not differentiable at a=b)

**Example:**
```python
op = ScalarMaxOp()
result = op.execute(3.0, 5.0)  # 5.0
```

#### `ScalarMinOp`

**Description:** Minimum of two scalars

**Signature:** `(scalar, scalar) -> scalar`

**Formula:** `min(a, b)`

**Differentiable:** No (not differentiable at a=b)

**Example:**
```python
op = ScalarMinOp()
result = op.execute(3.0, 5.0)  # 3.0
```

### Unary Operations

#### `ScalarAbsOp`

**Description:** Absolute value

**Signature:** `(scalar) -> scalar`

**Formula:** `|a|`

**Differentiable:** No (not differentiable at a=0)

**Example:**
```python
op = ScalarAbsOp()
result = op.execute(-5.0)  # 5.0
```

#### `ScalarNegOp`

**Description:** Negation

**Signature:** `(scalar) -> scalar`

**Formula:** `-a`

**Differentiable:** Yes

**Example:**
```python
op = ScalarNegOp()
result = op.execute(5.0)  # -5.0
```

---

## Vector Operations

All vector operations work with `np.ndarray` (1D arrays).

### Element-wise Operations

#### `VectorAddOp`

**Description:** Element-wise vector addition

**Signature:** `(vector, vector) -> vector`

**Formula:** `a + b` (element-wise)

**Differentiable:** Yes

**Example:**
```python
op = VectorAddOp()
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
result = op.execute(v1, v2)  # [5, 7, 9]
```

#### `VectorSubOp`

**Description:** Element-wise vector subtraction

**Signature:** `(vector, vector) -> vector`

**Formula:** `a - b` (element-wise)

**Differentiable:** Yes

#### `VectorMulOp`

**Description:** Element-wise vector multiplication

**Signature:** `(vector, vector) -> vector`

**Formula:** `a * b` (element-wise)

**Differentiable:** Yes

### Reduction Operations (Vector → Scalar)

#### `VectorDotProductOp`

**Description:** Dot product of two vectors

**Signature:** `(vector, vector) -> scalar`

**Formula:** `a · b = Σ(aᵢ * bᵢ)`

**Differentiable:** Yes

**Example:**
```python
op = VectorDotProductOp()
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
result = op.execute(v1, v2)  # 32 (1*4 + 2*5 + 3*6)
```

#### `VectorMeanOp`

**Description:** Mean of vector elements

**Signature:** `(vector) -> scalar`

**Formula:** `mean(v) = (1/n) * Σvᵢ`

**Differentiable:** Yes

**Example:**
```python
op = VectorMeanOp()
v = np.array([1, 2, 3, 4, 5])
result = op.execute(v)  # 3.0
```

#### `VectorMaxOp`

**Description:** Maximum of vector elements

**Signature:** `(vector) -> scalar`

**Formula:** `max(v)`

**Differentiable:** No (not differentiable at equality points)

#### `VectorMinOp`

**Description:** Minimum of vector elements

**Signature:** `(vector) -> scalar`

**Formula:** `min(v)`

**Differentiable:** No (not differentiable at equality points)

#### `VectorSumOp`

**Description:** Sum of vector elements

**Signature:** `(vector) -> scalar`

**Formula:** `sum(v) = Σvᵢ`

**Differentiable:** Yes

#### `VectorNormOp`

**Description:** L2 norm of vector

**Signature:** `(vector) -> scalar`

**Formula:** `||v|| = √(Σvᵢ²)`

**Differentiable:** Yes (differentiable everywhere except at v=0, but commonly used in practice)

---

## Matrix Operations

All matrix operations work with `np.ndarray` (2D arrays).

### Element-wise Operations

#### `MatrixAddOp`

**Description:** Element-wise matrix addition

**Signature:** `(matrix, matrix) -> matrix`

**Formula:** `A + B` (element-wise)

**Differentiable:** Yes

#### `MatrixSubOp`

**Description:** Element-wise matrix subtraction

**Signature:** `(matrix, matrix) -> matrix`

**Formula:** `A - B` (element-wise)

**Differentiable:** Yes

#### `MatrixMulOp`

**Description:** Matrix multiplication

**Signature:** `(matrix, matrix) -> matrix`

**Formula:** `A @ B` (matrix multiplication)

**Differentiable:** Yes

**Note:** Standard matrix multiplication, not element-wise.

### Reduction Operations (Matrix → Scalar)

#### `MatrixMeanOp`

**Description:** Global mean of matrix

**Signature:** `(matrix) -> scalar`

**Formula:** `mean(M) = (1/(h*w)) * ΣMᵢⱼ`

**Differentiable:** Yes

#### `MatrixMaxOp`

**Description:** Global maximum of matrix

**Signature:** `(matrix) -> scalar`

**Formula:** `max(M)`

**Differentiable:** No (not differentiable at equality points)

#### `MatrixMinOp`

**Description:** Global minimum of matrix

**Signature:** `(matrix) -> scalar`

**Formula:** `min(M)`

**Differentiable:** No (not differentiable at equality points)

### Reduction Operations (Matrix → Vector)

#### `MatrixRowMeanOp`

**Description:** Mean across rows (axis=0)

**Signature:** `(matrix) -> vector`

**Formula:** Mean of each column

**Differentiable:** Yes

**Example:**
```python
# For matrix [[1, 2], [3, 4]]
# Result: [2.0, 3.0] (mean of column 0, mean of column 1)
```

#### `MatrixColMeanOp`

**Description:** Mean across columns (axis=1)

**Signature:** `(matrix) -> vector`

**Formula:** Mean of each row

**Differentiable:** Yes

**Example:**
```python
# For matrix [[1, 2], [3, 4]]
# Result: [1.5, 3.5] (mean of row 0, mean of row 1)
```

#### `MatrixRowMaxOp`

**Description:** Max across rows (axis=0)

**Signature:** `(matrix) -> vector`

**Formula:** Max of each column

**Differentiable:** No (not differentiable at equality points)

#### `MatrixColMaxOp`

**Description:** Max across columns (axis=1)

**Signature:** `(matrix) -> vector`

**Formula:** Max of each row

**Differentiable:** No (not differentiable at equality points)

### Shape Operations

#### `MatrixFlattenOp`

**Description:** Flatten matrix to vector

**Signature:** `(matrix) -> vector`

**Formula:** `flatten(M)` (row-major order)

**Differentiable:** Yes

**Example:**
```python
# For matrix [[1, 2], [3, 4]]
# Result: [1, 2, 3, 4]
```

#### `MatrixTransposeOp`

**Description:** Transpose matrix

**Signature:** `(matrix) -> matrix`

**Formula:** `M^T`

**Differentiable:** Yes

**Example:**
```python
# For matrix [[1, 2], [3, 4]]
# Result: [[1, 3], [2, 4]]
```

---

## Cross-Type Operations

Operations that work across different memory types.

#### `ScalarVectorMulOp`

**Description:** Multiply vector by scalar (broadcast)

**Signature:** `(scalar, vector) -> vector`

**Formula:** `s * v` (each element multiplied by scalar)

**Differentiable:** Yes

**Example:**
```python
op = ScalarVectorMulOp()
result = op.execute(2.0, np.array([1, 2, 3]))  # [2, 4, 6]
```

#### `ScalarVectorAddOp`

**Description:** Add scalar to all vector elements (broadcast)

**Signature:** `(scalar, vector) -> vector`

**Formula:** `s + v` (scalar added to each element)

**Differentiable:** Yes

**Example:**
```python
op = ScalarVectorAddOp()
result = op.execute(1.0, np.array([1, 2, 3]))  # [2, 3, 4]
```

#### `ScalarMatrixMulOp`

**Description:** Multiply matrix by scalar (broadcast)

**Signature:** `(scalar, matrix) -> matrix`

**Formula:** `s * M` (each element multiplied by scalar)

**Differentiable:** Yes

#### `ScalarMatrixAddOp`

**Description:** Add scalar to all matrix elements (broadcast)

**Signature:** `(scalar, matrix) -> matrix`

**Formula:** `s + M` (scalar added to each element)

**Differentiable:** Yes

---

## Operation Registry

The module provides predefined lists of operation classes for easy access:

### Lists

- **`SCALAR_OPS`**: All scalar operation classes
- **`VECTOR_OPS`**: All vector operation classes
- **`MATRIX_OPS`**: All matrix operation classes
- **`CROSS_TYPE_OPS`**: All cross-type operation classes
- **`ALL_OPS`**: All operations combined

### Helper Functions

#### `get_operations_by_output_type(output_type: MemoryType) -> List[type]`

Get all operation classes that produce a given output type.

**Parameters:**
- `output_type`: The desired output type

**Returns:**
- `List[type]`: List of operation classes

**Example:**
```python
scalar_ops = get_operations_by_output_type(MemoryType.SCALAR)
# Returns all operations that output scalars
```

#### `get_operations_by_input_types(input_types: List[MemoryType]) -> List[type]`

Get all operation classes that take specific input types.

**Parameters:**
- `input_types`: List of input types

**Returns:**
- `List[type]`: List of operation classes

**Example:**
```python
ops = get_operations_by_input_types([MemoryType.SCALAR, MemoryType.SCALAR])
# Returns all operations that take two scalars as input
```

---

## Usage Examples

### Creating and Using Operations

```python
from operation import ScalarAddOp, VectorDotProductOp, MatrixMeanOp

# Create operation instances
add_op = ScalarAddOp()
dot_op = VectorDotProductOp()
mean_op = MatrixMeanOp()

# Execute operations
result1 = add_op.execute(3.0, 2.0)  # 5.0
result2 = dot_op.execute(np.array([1, 2]), np.array([3, 4]))  # 11.0
result3 = mean_op.execute(np.array([[1, 2], [3, 4]]))  # 2.5

# Check operation properties
print(add_op.name)  # "scalar_add"
print(add_op.input_types())  # [MemoryType.SCALAR, MemoryType.SCALAR]
print(add_op.output_type())  # MemoryType.SCALAR
print(add_op.differentiable)  # True
```

### Using Operation Registry

```python
from operation import ALL_OPS, SCALAR_OPS, get_operations_by_output_type

# Get all operations
print(f"Total operations: {len(ALL_OPS)}")

# Get all scalar operations
print(f"Scalar operations: {len(SCALAR_OPS)}")

# Get operations that produce scalars
scalar_producing = get_operations_by_output_type(MemoryType.SCALAR)
print(f"Operations producing scalars: {len(scalar_producing)}")
```

### Creating Random Operations

```python
import numpy as np
from operation import ALL_OPS

# Instantiate random operation
rng = np.random.default_rng()
op_class = rng.choice(ALL_OPS)
op = op_class()

print(f"Random operation: {op}")
print(f"Input types: {op.input_types()}")
print(f"Output type: {op.output_type()}")
```

---

## Design Notes

### Type Safety

All operations are type-safe. The type system ensures:
- Operations can only be used with compatible memory types
- Instructions are validated for type compatibility
- Type mismatches are caught at instruction creation time

### Differentiability

The `differentiable` property indicates whether an operation is differentiable:
- **True**: Operation is differentiable (can be used in gradient-based optimization)
- **False**: Operation has discontinuities or non-differentiable points

This is useful for:
- Automatic differentiation
- Gradient-based learning
- Understanding operation behavior

### Immutability

Operations are stateless and immutable. A single operation instance can be reused across multiple instructions, which is memory-efficient.

### Extensibility

To add a new operation:
1. Create a new class inheriting from `Operation`
2. Implement all abstract methods
3. Add the class to the appropriate list (`SCALAR_OPS`, `VECTOR_OPS`, etc.)

---

## Operation Summary Table

| Category | Count | Operations |
|----------|-------|------------|
| Scalar | 8 | add, sub, mul, div_protected, max, min, abs, neg |
| Vector | 9 | add, sub, mul, dot, mean, max, min, sum, norm |
| Matrix | 12 | add, sub, mul, mean, max, min, row_mean, col_mean, row_max, col_max, flatten, transpose |
| Cross-Type | 4 | scalar_vector_mul, scalar_vector_add, scalar_matrix_mul, scalar_matrix_add |
| **Total** | **33** | |

---

## Notes

- All operations use NumPy for array operations
- Vector operations assume 1D arrays
- Matrix operations assume 2D arrays
- Operations are stateless and can be reused
- Type safety is enforced at the instruction level
- All data is handled as `np.float32` for consistency

