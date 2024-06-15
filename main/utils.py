import random
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
import io
import base64

import scipy.signal


polynomes_dict = {
    1: [''],
    2: ['<1 7H>'],
    3: ['<1 13F>'],
    4: ['<1 23F>', '<3 37D>'],
    5: ['<1 45E>', '<3 75G>', '<5 67H>'],
    6: ['<1 103F>', '<3 127B>', '<5 147H>', '<7 111A>', '<11 155E>'],
    7: ['<11 325G>', '<3 217E>', '<5 235E>', '<13 203F>'],
    8: ['<1 435E>', '<3 567B>', '<5 763D>', '<7 551E>', '<9 675C>'],
    9: ['<1 1021E>', '<3 1131E>', '<5 1461G>', '<7 1231A>', '<9 1423G>', '<11 1055E>'],
    10: ['<1 2011E>', '<3 2017B>', '<5 2415E>', '<7 3771G>', '<9 2257B>', '<11 2065A>'],
    11: ['<1 4005E>', '<3 4445E>', '<5 4215E>', '<9 6015G>', '<11 7413H>', '<13 4143F>'],
    12: ['<1 10123F>', '<3 12133B>', '<5 10115A>', '<7 12153B>', '<9 11765A>'],
    13: ['<1 20033F>', '<3 23261E>', '<5 24623F>', '<7 23517F>', '<9 30741G>'],
    14: ['<1 47103F>', '<3 40547B>', '<5 43333E>', '<7 51761E>', '<9 54055A>'],
    15: ['<1 100003F>', '<3 102043F>', '<5 110013F>', '<7 125253B>', '<9 102067F>'],
}


def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


def calc_t(j, n):
    return int((2**n)-1/gcd((2**n)-1, j))


def calculate_hemming_weight(r, n, m):
    return ((2 ** r) - 1) * 2 ** (m + n - r - 1)


def create_struct_matrix_var_1(a, pol):
    matrix = np.zeros((a, a), dtype=int)
    matrix[0] = pol

    mod_matrix = matrix[1:, :]

    for i in range(len(mod_matrix)):
        for j in range(len(mod_matrix[i])):
            if i == j:
                mod_matrix[i][j] = 1

    return matrix


def create_struct_matrix_var_2(a, pol):
    matrix = np.zeros((a, a), dtype=int)
    matrix[:, 0] = pol

    mod_matrix = matrix[:, 1:]

    for i in range(len(mod_matrix)):
        for j in range(len(mod_matrix[i])):
            if i == j:
                mod_matrix[i][j] = 1

    return matrix


def create_S_matrix(r, n, m):
    matrix = np.zeros((n, m), dtype=int)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if i == j and r != 0:
                matrix[i][j] = 1
                r -= 1
    return matrix


def create_sequence(i, j, A, B, S):
    sequence = []
    states = []

    S_0 = S.copy()

    V = np.mod(np.dot(A, S), 2)
    S_Tnext = np.mod(np.dot(V, B), 2)

    sequence.append(S[i - 1, j - 1])
    states.append(S_Tnext)

    while not np.array_equal(S_0, S_Tnext):
        sequence.append(S_Tnext[i - 1, j - 1])
        V = np.mod(np.dot(A, S_Tnext), 2)
        S_Tnext = np.mod(np.dot(V, B), 2)
        states.append(S_Tnext)

    return sequence, states



def get_polynomial(str_pol):
    space_index = str_pol.find(' ')
    number_str = str_pol[space_index + 1:len(str_pol) - 2]

    number = int(number_str, 8)
    number_binary = bin(number)[3:]
    pol = list(map(int, list(number_binary)))
    return pol


def get_states(pol):
    states = [*range(0, 2 ** len(pol))]
    for i, state in enumerate(states):
        states[i] = list(map(int, bin(state)[2:].zfill(len(pol))))

    return states


def generate_states(states, state_matrix, polynom_coefficients):
    j = 0
    res = []
    for state in state_matrix:
        states.remove(state)
        j += 1

        result = 0
        next_state = []
        for i, c in enumerate(polynom_coefficients):
            if c & 1:
                result ^= state[i]

        next_state.append(result & 1)
        next_state.extend(state[:-1])
        res.append((j, state, int(''.join(map(str, state)), 2)))

        if next_state in states:
            state_matrix.append(next_state)
        else:
            state_matrix.clear()
            if len(states):
                state_matrix.append(random.choice(states))

            return state_matrix, j, res


def normalize_acf(acf, sequence):
    acf = acf / np.sum(sequence ** 2)
    middle_index = len(acf) // 2
    first_half = acf[:middle_index]
    second_half = acf[middle_index:]

    mirrored_first_half = first_half[::-1]
    mirrored_second_half = second_half[::-1]
    max_value = np.max(acf)
    mirrored_first_half = np.insert(mirrored_first_half, 0, max_value)
    acf = np.concatenate((mirrored_first_half, mirrored_second_half))
    return acf


def normalize_seq(seq):
    for i in range(len(seq)):
        if not seq[i]:
            seq[i] = 1
        else:
            seq[i] = -1
    return np.array(seq)

def m_autocorrelation(data):
  acf = []
  T = len(data)
  coeficient = 1.0 / T

  for tau in range(T + 1):
    sum = 0.0
    for i in range(T):
      a = 1.0 if data[i] == 0 else -1.0
      b = 1.0 if data[(i + tau) % T] == 0 else -1.0
      sum += a * b

    acf.append(coeficient * sum)

  return acf

def generate_acf_image(sequence):
    acf = m_autocorrelation(sequence)

    lags = np.arange(len(acf))
    plt.clf()
    min_value = np.min(acf)

    plt.plot(lags, acf, color='tab:green', linewidth=0.5)

    min_text = f'min value: {min_value:.4f}'
    plt.annotate(min_text, xy=(len(lags) / 2, 1),
                 xytext=(0, 5), textcoords='offset points',
                 fontsize=10, ha='center', va='top')

    plt.grid(True)

    plt.xlabel('Період')
    plt.ylabel('Нормалізована АКФ')
    plt.title('АКФ бінарної послідовності')

    # Сохраняем график как изображение в буфере
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Преобразуем изображение в строку base64
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    return image_base64
def generate_matrix_acf_image(sequence):
    sequence = normalize_seq(sequence)
    acf = np.correlate(sequence, sequence, mode='same')

    acf_normalized = normalize_acf(acf, sequence)
    # acf_normalized = clip_negative_values(acf_normalized)
    lags = np.arange(len(acf_normalized))

    plt.clf()


    plt.plot(lags, acf_normalized, color='tab:green', linewidth=0.5)

    plt.grid(True)

    plt.xlabel('Період')
    plt.ylabel('Нормалізована АКФ')
    plt.title('АКФ бінарної послідовності')

    # Сохраняем график как изображение в буфере
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Преобразуем изображение в строку base64
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    return image_base64


def generate_torus(TA, TB, arrA, arrB, S0):
    torus = []
    for i in range(TA - 1, -1, -1):
        for j in range(TB - 1, -1, -1):
            A = np.linalg.matrix_power(arrA, i) % 2
            B = np.linalg.matrix_power(arrB, j) % 2

            V = np.matmul(A, S0) % 2
            state = np.matmul(V, B) % 2
            torus.append(state)
    return torus

def normalize_autocorr(autocorr_matrix):
    n_rows, n_cols = autocorr_matrix.shape

    normalization_factor = 1 / (n_rows * n_cols)
    norm_autocorr = autocorr_matrix * normalization_factor * 4.0
    return norm_autocorr


def autocorrelation(matrix):
    matrix = np.array(matrix)
    n_rows, n_cols = matrix.shape
    autocorr_matrix = np.zeros_like(matrix, dtype=float)

    normalization_factor = 1 / (n_rows * n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            sub_matrix = matrix[max(0, -i):n_rows - max(0, i), max(0, -j):n_cols - max(0, j)]
            autocorr_matrix[i, j] = normalization_factor * np.sum(matrix[max(0, i):n_rows - max(0, -i), max(0, j):n_cols - max(0, -j)] * sub_matrix)

    return autocorr_matrix

def generate_two_dim_acf_image(TA, TB, S0, torus, method, mode = 0):
    S_num_rows, S_num_cols = S0.shape
    large_array = np.concatenate([arr.flatten() for arr in torus])
    large_array = normalize_seq(large_array)
    num_matrices = len(large_array) // (TA * S_num_rows * TB * S_num_cols)
    reshaped_matrices = large_array.reshape((num_matrices, TA * S_num_rows, TB * S_num_cols))

    if mode:
        xcorr = method(reshaped_matrices[0])
    else:
        xcorr = method(reshaped_matrices[0], reshaped_matrices[0])
        xcorr = normalize_autocorr(xcorr)


    plt.clf()

    plt.imshow(xcorr, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Autocorrelation of state torus ')
    plt.title('2D Autocorrelation')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    return image_base64



def generate_two_dim_acf_image_min(sequence, states, S):
    large_arr =[]
    for state in states:
        large_arr.extend(state[0])
        large_arr.extend(state[1:, 0])


    f, seq_copy, n, m = factorize(large_arr)
    large_arr = normalize_seq(large_arr)
    result_matrix = create_pvt_matrix_var_2(large_arr, n, m)

    xcorr = autocorrelation(result_matrix)


    plt.clf()

    plt.imshow(xcorr, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Autocorrelation of state min torus ')
    plt.title('2D Autocorrelation')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    return image_base64


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def factorize(sequence):
    value = len(sequence)
    f = 0
    if is_prime(value):
        value += 1
        sequence.append(1)
        f = 1

    res = []
    for x in range(1, int(sqrt(value)) + 1):
        if value % x == 0:
            res.append([x, value // x])
    return f, sequence, *res[-1]



def create_pvt_matrix(seq, n, m):
    matrix = np.zeros((n, m))

    i, j = 0, 0
    for item in seq:
        matrix[i][j] = item
        i = (i + 1) % n  # Move down one row
        j = (j + 1) % m  # Move right one column

    return matrix


def create_pvt_matrix_var_2(seq, n, m):
    seq_array = np.array(seq)
    matrix = seq_array.reshape((n, m))
    return matrix

def create_pvt_matrix1(seq):
    seq_len = len(seq)
    # Calculate the dimensions of the matrix
    n = int(np.ceil(np.sqrt(seq_len)))
    m = int(np.ceil(seq_len / n))
    # Initialize the matrix with zeros
    matrix = np.zeros((n, m))
    # Fill the matrix with the sequence
    i, j = 0, 0
    for item in seq:
        matrix[i][j] = item

        # Move down one row and right one column
        i = (i + 1) % n
        j = (j + 1) % m
    return np.array(matrix)


def generate_pvt_acf_image(pvt_matrix):

    xcorr = scipy.signal.correlate2d(pvt_matrix, pvt_matrix)
    xcorr = normalize_autocorr(xcorr)
    plt.clf()

    plt.imshow(xcorr, cmap='coolwarm', interpolation='nearest') #coolwarm binary twilight_shifted
    plt.colorbar(label='Autocorrelation of PRA ')
    plt.title('2D Autocorrelation PRA')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    return image_base64
