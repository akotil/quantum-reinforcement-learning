import numpy as np
import scipy
from scipy import linalg
from nptyping import NDArray, Float
from typing import Any

def rescale_eigenvalues(A, b):
    mul_factor = 0.5
    multiplied = 0
    eigenvalues = linalg.eigvals(A)

    # λ := λ * mul_factor
    while np.any(eigenvalues >= 0.5):
        A *= mul_factor
        eigenvalues = linalg.eigvals(A)
        multiplied += 1

    # b must be a unit vector if no scaling is present. Otherwise, b is scaled by the mul_factor
    # to keep the solution vector identical to the original problem.
    if multiplied == 0:
        scaled = False
        b = b / np.linalg.norm(b)
    else:
        scaled = True
        b *= (mul_factor ** multiplied)
    return A, b, scaled


def prepare_hhl(A, b):
    # Is A Hermitian?
    if np.allclose(A.conj().T, A):
        transformed = False
        print("A is Hermitian. Proceeding.\n")
        print("A:\n", A)
    else:
        transformed = True
        # A is not Hermitian, applying the method from the paper
        print("A is not Hermitian. Transforming A.\n")
        n = A.shape[0]
        A_trans = np.zeros((n * 2, n * 2), dtype="complex")
        A_trans[:n, n:] = A
        A_trans[n:, :n] = A.conj().T

        b_trans = np.pad(b, (0, n), 'constant')

        A = A_trans
        b = b_trans

    A, b, scaled = rescale_eigenvalues(A, b)
    return A, b, transformed, scaled


def hhl(A: NDArray[(Any, Any), Float], b: NDArray[Any, Float], epsilon: Float, T: int) -> NDArray[Any, Float]:
    A, b, transformed, scaled = prepare_hhl(A, b)

    # Calculate the singular values of A to set k
    s = scipy.linalg.svd(A, compute_uv=False)
    k = s[0] / s[-1]
    print('condition number k: ', k)

    # Calculate basis coefficients for phi_0 and the registers containing phi_0 and b
    phi_0 = [np.sqrt(2 / T) * np.sin(np.pi * (i + 1 / 2) / T)
             for i in range(T)]
    registers = np.kron(phi_0, b)

    # Hamiltonian Evolution
    n = A.shape[0]
    H = np.zeros((T, n, n), dtype='complex')
    t_0 = k / epsilon
    for i in range(T):
        H[i] = linalg.expm(1j * A * i * t_0 / T)

    # Apply Hamiltonian Evolution to the input registers
    state = [np.dot(H[i], registers.reshape((T, n))[i]) for i in range(T)]

    # Apply Fourier Transformation to the first register
    state = np.fft.fft(state, axis=0, norm='ortho')

    # Conditioned rotation on the ancilla qubit
    C = 0.1 / k
    c1 = np.zeros(T, dtype='complex')
    one_state = np.zeros((T, n), dtype='complex')
    for i in range(0, T):
        if not transformed:
            eigenvalue = 2 * np.pi * i / t_0
        else:
            eigenvalue = 2 * np.pi * i / t_0 if i < T // 2 else 2 * np.pi * (i - T) / t_0
        c1[i] = C / eigenvalue if C <= abs(eigenvalue) else 0
        one_state[i] = state[i] * c1[i]

    # Inverse Fourier Transformation
    one_state = np.fft.ifft(one_state, axis=0, norm='ortho')

    # Inverse Hamiltonian Evaluation
    one_state = [np.dot(H[i].conj().T, one_state[i]) for i in range(T)]

    # Inverse phi_0 by reversing the Kronecker Product
    one_state /= C
    solution = one_state[T // 2] / phi_0[T // 2]

    if transformed:
        return solution[n // 2:]
    else:
        return solution


def main():
    n = 4

    b = np.random.random(n)
    A = np.random.random((n, n))

    T = 5000
    epsilon = 0.001
    x = hhl(A, b, epsilon, T)

    print('x:\n', x)

    # classical solution
    x_cl = np.linalg.solve(A, b)
    print('xref:\n', x_cl)


if __name__ == '__main__':
    main()
