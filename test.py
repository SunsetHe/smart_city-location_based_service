import numpy as np


def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    Args:
        A (numpy.ndarray): Symmetric positive-definite matrix.
    Returns:
        L (numpy.ndarray): Lower triangular matrix such that A = L * L.T.
    """
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            if i == j:  # Diagonal elements
                L[i, j] = np.sqrt(A[i, i] - np.sum(L[i, :j] ** 2))
            else:  # Off-diagonal elements
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

    return L


# Input the matrix A
print("Enter the dimension of the matrix (n):")
n = int(input())
print(f"Enter the {n}x{n} matrix row by row (space-separated):")
A = np.array([list(map(float, input().split())) for _ in range(n)])

# Compute B = A * A.T
B = A.T @ A
print("Matrix B = A.T * A:")
print(B)

# Perform Cholesky decomposition on B
try:
    L = cholesky_decomposition(B)
    print("Lower triangular matrix L (Cholesky decomposition of B):")
    print(L)
    print("Reconstructed matrix B (L * L.T):")
    print(L @ L.T)
except Exception as e:
    print("Error during Cholesky decomposition:", e)

import math

def to_sqrt_fraction(x, tolerance=1e-7):
    # 搜索合理的整数 a, b，使 x ≈ a / sqrt(b)
    for b in range(1, 101):  # 限定 b 的范围
        a = round(x * math.sqrt(b))
        if abs(x - a / math.sqrt(b)) < tolerance:
            return f"{a}/sqrt({b})"
    return f"{x} (无法简化)"

# 浮点数列表
numbers = [3.74165739, 2.39045722, 1.67332005, 0.4472136]

# 转换为根号分数形式
sqrt_fractions = [to_sqrt_fraction(num) for num in numbers]

# 输出结果
print("根号分数表示:")
print(sqrt_fractions)
