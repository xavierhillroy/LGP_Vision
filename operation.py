import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Any
from memory_system import MemoryType

class Operation(ABC):
    """Abstract class for all operations 
    Each operations know its input and output types and how to execute it"""
    @abstractmethod
    def input_types(self) -> List[MemoryType]:
        """Return the input types of the operation"""
        pass

    @abstractmethod
    def output_type(self) -> MemoryType:
        """Return the output type of the operation"""
        pass

    @abstractmethod
    def execute(self, inputs: List[Any]) -> Any:
        """Execute the operation"""
        pass
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the operation"""
        pass
    @property
    @abstractmethod
    def differentiable(self) -> bool:
        """Return if the operation is differentiable"""
        pass
    def __repr__(self) -> str:
        """Return the string representation of the operation"""
        input_str = ", ".join([t.value for t in self.input_types()])
        return f"{self.name}({input_str}) -> {self.output_type().value}"
    #============================= SCALAR OPERATIONS =============================
class ScalarAddOp(Operation):
    """Add two scalar values"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    def execute(self, a: float, b: float) -> float:   
        return a + b
    @property
    def name(self) -> str:
        return "scalar_add"
    @property
    def differentiable(self) -> bool:
        return True

class ScalarSubOp(Operation):
    """Subtract two scalar values"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    def execute(self, a: float, b: float) -> float:   
        return a - b
    @property
    def name(self) -> str:
        return "scalar_sub"
    @property
    def differentiable(self) -> bool:
        return True
    
class ScalarMulOp(Operation):
    """Multiply two scalars: a * b"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        return a * b
    
    @property
    def name(self) -> str:
        return "scalar_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class ScalarDivProtectedOp(Operation):
    """Protected division: a / b (returns 1.0 if b == 0)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        return a / b if abs(b) > 1e-8 else 1.0
    
    @property
    def name(self) -> str:
        return "scalar_div_protected"
    
    @property
    def differentiable(self) -> bool:
        return False  # Has discontinuity at b=0
class ScalarMaxOp(Operation):
    """Maximum of two scalars: max(a, b)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        return max(a, b)
    
    @property
    def name(self) -> str:
        return "scalar_max"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at a=b


class ScalarMinOp(Operation):
    """Minimum of two scalars: min(a, b)"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float, b: float) -> float:
        return min(a, b)
    
    @property
    def name(self) -> str:
        return "scalar_min"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at a=b


class ScalarAbsOp(Operation):
    """Absolute value: |a|"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        return abs(a)
    
    @property
    def name(self) -> str:
        return "scalar_abs"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at a=0


class ScalarNegOp(Operation):
    """Negation: -a"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: float) -> float:
        return -a
    
    @property
    def name(self) -> str:
        return "scalar_neg"
    
    @property
    def differentiable(self) -> bool:
        return True
# ==================== VECTOR OPERATIONS ====================

class VectorAddOp(Operation):
    """Element-wise vector addition: a + b"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b
    
    @property
    def name(self) -> str:
        return "vector_add"
    
    @property
    def differentiable(self) -> bool:
        return True


class VectorSubOp(Operation):
    """Element-wise vector subtraction: a - b"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b
    
    @property
    def name(self) -> str:
        return "vector_sub"
    
    @property
    def differentiable(self) -> bool:
        return True


class VectorMulOp(Operation):
    """Element-wise vector multiplication: a * b"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b
    
    @property
    def name(self) -> str:
        return "vector_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class VectorDotProductOp(Operation):
    """Dot product of two vectors: a · b → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
    
    @property
    def name(self) -> str:
        return "vector_dot"
    
    @property
    def differentiable(self) -> bool:
        return True


class VectorMeanOp(Operation):
    """Mean of vector elements: mean(v) → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, v: np.ndarray) -> float:
        return float(np.mean(v))
    
    @property
    def name(self) -> str:
        return "vector_mean"
    
    @property
    def differentiable(self) -> bool:
        return True


class VectorMaxOp(Operation):
    """Maximum of vector elements: max(v) → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, v: np.ndarray) -> float:
        return float(np.max(v))
    
    @property
    def name(self) -> str:
        return "vector_max"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class VectorMinOp(Operation):
    """Minimum of vector elements: min(v) → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, v: np.ndarray) -> float:
        return float(np.min(v))
    
    @property
    def name(self) -> str:
        return "vector_min"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class VectorSumOp(Operation):
    """Sum of vector elements: sum(v) → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, v: np.ndarray) -> float:
        return float(np.sum(v))
    
    @property
    def name(self) -> str:
        return "vector_sum"
    
    @property
    def differentiable(self) -> bool:
        return True


class VectorNormOp(Operation):
    """L2 norm of vector: ||v|| → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, v: np.ndarray) -> float:
        return float(np.linalg.norm(v))
    
    @property
    def name(self) -> str:
        return "vector_norm"
    
    @property
    def differentiable(self) -> bool:
        return True  # Differentiable everywhere except at v=0 (but commonly used in practice)
# ==================== MATRIX OPERATIONS ====================

class MatrixAddOp(Operation):
    """Element-wise matrix addition: A + B"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b
    
    @property
    def name(self) -> str:
        return "matrix_add"
    
    @property
    def differentiable(self) -> bool:
        return True


class MatrixSubOp(Operation):
    """Element-wise matrix subtraction: A - B"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b
    
    @property
    def name(self) -> str:
        return "matrix_sub"
    
    @property
    def differentiable(self) -> bool:
        return True


class MatrixMulOp(Operation):
    """Matrix multiplication: A @ B"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)
    
    @property
    def name(self) -> str:
        return "matrix_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class MatrixMeanOp(Operation):
    """Global mean of matrix: mean(M) → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, m: np.ndarray) -> float:
        return float(np.mean(m))
    
    @property
    def name(self) -> str:
        return "matrix_mean"
    
    @property
    def differentiable(self) -> bool:
        return True


class MatrixMaxOp(Operation):
    """Global maximum of matrix: max(M) → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, m: np.ndarray) -> float:
        return float(np.max(m))
    
    @property
    def name(self) -> str:
        return "matrix_max"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class MatrixMinOp(Operation):
    """Global minimum of matrix: min(M) → scalar"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.SCALAR
    
    def execute(self, m: np.ndarray) -> float:
        return float(np.min(m))
    
    @property
    def name(self) -> str:
        return "matrix_min"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class MatrixRowMeanOp(Operation):
    """Mean across rows (axis=0): M → vector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        return np.mean(m, axis=0)
    
    @property
    def name(self) -> str:
        return "matrix_row_mean"
    
    @property
    def differentiable(self) -> bool:
        return True


class MatrixColMeanOp(Operation):
    """Mean across columns (axis=1): M → vector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        return np.mean(m, axis=1)
    
    @property
    def name(self) -> str:
        return "matrix_col_mean"
    
    @property
    def differentiable(self) -> bool:
        return True


class MatrixRowMaxOp(Operation):
    """Max across rows (axis=0): M → vector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        return np.max(m, axis=0)
    
    @property
    def name(self) -> str:
        return "matrix_row_max"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class MatrixColMaxOp(Operation):
    """Max across columns (axis=1): M → vector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        return np.max(m, axis=1)
    
    @property
    def name(self) -> str:
        return "matrix_col_max"
    
    @property
    def differentiable(self) -> bool:
        return False  # Not differentiable at equality points


class MatrixFlattenOp(Operation):
    """Flatten matrix to vector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        return m.flatten()
    
    @property
    def name(self) -> str:
        return "matrix_flatten"
    
    @property
    def differentiable(self) -> bool:
        return True


class MatrixTransposeOp(Operation):
    """Transpose matrix: M^T"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, m: np.ndarray) -> np.ndarray:
        return m.T
    
    @property
    def name(self) -> str:
        return "matrix_transpose"
    
    @property
    def differentiable(self) -> bool:
        return True


# ==================== CROSS-TYPE OPERATIONS ====================

class ScalarVectorMulOp(Operation):
    """Multiply vector by scalar: s * v → vector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, s: float, v: np.ndarray) -> np.ndarray:
        return s * v
    
    @property
    def name(self) -> str:
        return "scalar_vector_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class ScalarVectorAddOp(Operation):
    """Add scalar to all vector elements: s + v → vector"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.VECTOR]
    
    def output_type(self) -> MemoryType:
        return MemoryType.VECTOR
    
    def execute(self, s: float, v: np.ndarray) -> np.ndarray:
        return s + v
    
    @property
    def name(self) -> str:
        return "scalar_vector_add"
    
    @property
    def differentiable(self) -> bool:
        return True


class ScalarMatrixMulOp(Operation):
    """Multiply matrix by scalar: s * M → matrix"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, s: float, m: np.ndarray) -> np.ndarray:
        return s * m
    
    @property
    def name(self) -> str:
        return "scalar_matrix_mul"
    
    @property
    def differentiable(self) -> bool:
        return True


class ScalarMatrixAddOp(Operation):
    """Add scalar to all matrix elements: s + M → matrix"""
    def input_types(self) -> List[MemoryType]:
        return [MemoryType.SCALAR, MemoryType.MATRIX]
    
    def output_type(self) -> MemoryType:
        return MemoryType.MATRIX
    
    def execute(self, s: float, m: np.ndarray) -> np.ndarray:
        return s + m
    
    @property
    def name(self) -> str:
        return "scalar_matrix_add"
    
    @property
    def differentiable(self) -> bool:
        return True

# ==================== OPERATION REGISTRY ====================

# All scalar operations
SCALAR_OPS = [
    ScalarAddOp,
    ScalarSubOp,
    ScalarMulOp,
    ScalarDivProtectedOp,
    ScalarMaxOp,
    ScalarMinOp,
    ScalarAbsOp,
    ScalarNegOp,
]

# All vector operations
VECTOR_OPS = [
    VectorAddOp,
    VectorSubOp,
    VectorMulOp,
    VectorDotProductOp,
    VectorMeanOp,
    VectorMaxOp,
    VectorMinOp,
    VectorSumOp,
    VectorNormOp,
]

# All matrix operations
MATRIX_OPS = [
    MatrixAddOp,
    MatrixSubOp,
    MatrixMulOp,
    MatrixMeanOp,
    MatrixMaxOp,
    MatrixMinOp,
    MatrixRowMeanOp,
    MatrixColMeanOp,
    MatrixRowMaxOp,
    MatrixColMaxOp,
    MatrixFlattenOp,
    MatrixTransposeOp,
]

# All cross-type operations
CROSS_TYPE_OPS = [
    ScalarVectorMulOp,
    ScalarVectorAddOp,
    ScalarMatrixMulOp,
    ScalarMatrixAddOp,
]

# All operations combined
ALL_OPS = SCALAR_OPS + VECTOR_OPS + MATRIX_OPS + CROSS_TYPE_OPS


def get_operations_by_output_type(output_type: MemoryType) -> List[type]:
    """Get all operation classes that produce a given output type"""
    return [op for op in ALL_OPS if op().output_type() == output_type]


def get_operations_by_input_types(input_types: List[MemoryType]) -> List[type]:
    """Get all operation classes that take specific input types"""
    return [op for op in ALL_OPS if op().input_types() == input_types]

if __name__ == "__main__":
    print("="*60)
    print("OPERATION TEST")
    print("="*60)
    
    # Test scalar operations
    print("\n--- Scalar Operations ---")
    add_op = ScalarAddOp()
    print(f"{add_op}: {add_op.execute(3.14, 2.71)}")
    
    div_op = ScalarDivProtectedOp()
    print(f"{div_op}: {div_op.execute(10.0, 0.0)} (protected)")
    
    # Test vector operations
    print("\n--- Vector Operations ---")
    v1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    v2 = np.array([5, 4, 3, 2, 1], dtype=np.float32)
    
    vadd_op = VectorAddOp()
    print(f"{vadd_op}: {vadd_op.execute(v1, v2)}")
    
    vdot_op = VectorDotProductOp()
    print(f"{vdot_op}: {vdot_op.execute(v1, v2)}")
    
    vmean_op = VectorMeanOp()
    print(f"{vmean_op}: {vmean_op.execute(v1)}")
    
    # Test matrix operations
    print("\n--- Matrix Operations ---")
    m1 = np.random.rand(4, 4).astype(np.float32)
    
    mmean_op = MatrixMeanOp()
    print(f"{mmean_op}: {mmean_op.execute(m1):.4f}")
    
    mrow_op = MatrixRowMeanOp()
    print(f"{mrow_op}: {mrow_op.execute(m1)}")
    
    # Test cross-type operations
    print("\n--- Cross-Type Operations ---")
    sv_op = ScalarVectorMulOp()
    print(f"{sv_op}: {sv_op.execute(2.0, v1)}")
    
    # Test registry
    print("\n--- Operation Registry ---")
    print(f"Total operations: {len(ALL_OPS)}")
    print(f"Scalar ops: {len(SCALAR_OPS)}")
    print(f"Vector ops: {len(VECTOR_OPS)}")
    print(f"Matrix ops: {len(MATRIX_OPS)}")
    print(f"Cross-type ops: {len(CROSS_TYPE_OPS)}")
    
    scalar_producing = get_operations_by_output_type(MemoryType.SCALAR)
    print(f"\nOperations that produce scalars: {len(scalar_producing)}")
    for op_class in scalar_producing[:5]:
        print(f"  - {op_class().name}")
    
    print("\n✅ All operations working!")
