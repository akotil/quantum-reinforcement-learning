import numpy as np
from scipy import linalg

SCALE_FACTOR = 0.0
B_NORM = 1.0


def rescale_eigenvalues(A):
    eigenvalues = linalg.eigvals(A)
    scaled = False
    if np.any(eigenvalues > 0.5):
        scaled = True
        max_eigval = max([abs(i) for i in linalg.eigvals(A)])
        A *= 1 / (2 * max_eigval)
        global SCALE_FACTOR
        SCALE_FACTOR = 1 / (2 * max_eigval)  # λ := λ * mul_factor

    print("smallest eigenvalue: ", min([abs(i) for i in linalg.eigvals(A)]))
    print("biggest eigenvalue: ", max([abs(i) for i in linalg.eigvals(A)]))
    return A, scaled


def prepare_hhl(A, b):
    # Is A Hermitian?
    if np.allclose(A.conj().T, A):
        transformed = False
        print("A is Hermitian. Proceeding.\n")
        print("A:\n", A)
    else:
        transformed = True
        print("A is not Hermitian. Transforming A.\n")
        n = A.shape[0]
        A_trans = np.zeros((n * 2, n * 2), dtype="complex")
        A_trans[:n, n:] = A
        A_trans[n:, :n] = A.conj().T

        b_trans = np.pad(b, (0, n), 'constant')

        A = A_trans
        b = b_trans

    global B_NORM
    B_NORM = np.linalg.norm(b)
    b = b / B_NORM  # b must be a unit vector
    A, scaled = rescale_eigenvalues(A)
    return A, b, transformed, scaled


def hhl(A, b, epsilon, T):
    """
    :param A: ndarray
            2d array containing the input matrix
    :param b: ndarray
            1d array containing the right hand side
    :param epsilon: float
            error constant
    :param T: int
            number of evolution time steps
    :return: ndarray
            1d array containing the solution vector
    """
    A, b, transformed, scaled = prepare_hhl(A, b)

    k = np.linalg.cond(A)

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
    one_state = np.zeros((T, n), dtype='complex')
    for i in range(T):
        if not scaled or (scaled and i < T // 2):
            eigenvalue = 2 * np.pi * i / t_0
        else:
            eigenvalue = 2 * np.pi * (i - T) / t_0

        c_1 = C / eigenvalue if C <= abs(eigenvalue) else 0
        one_state[i] = c_1 * state[i]

    # Inverse Fourier Transformation
    one_state = np.fft.ifft(one_state, axis=0, norm='ortho')

    # Inverse Hamiltonian Evolution
    one_state = [np.dot(H[i].conj().T, one_state[i]) for i in range(T)]

    # Inverse phi_0 by reversing the Kronecker Product
    one_state /= C
    solution = one_state[T // 2] / phi_0[T // 2]

    solution *= SCALE_FACTOR
    solution *= B_NORM

    if transformed:

        return solution[n // 2:]
    else:
        return solution


def main():
    n = 5
    b = np.random.random(n)
    A = np.random.random((n, n)) + 1j * np.random.random((n, n))
    T = 5000
    epsilon = 0.01
    x = hhl(A, b, epsilon, T)

    print('x:\n', x)
    residue = np.linalg.norm(np.dot(A, x) - b)
    print('residue: ', residue)

    # classical solution
    x_cl = np.linalg.solve(A, b)
    print('xref:\n', x_cl)


if __name__ == '__main__':
    main()
