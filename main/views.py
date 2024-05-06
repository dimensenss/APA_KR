import random
import re

import numpy as np
import matplotlib.pyplot as plt
import io
import urllib, base64
from scipy.signal import find_peaks

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.template.loader import render_to_string

polynomes_dict = {
    1: [''],
    2: ['<1 7H>'],
    3: ['<1 13F>'],
    4: ['<1 23F>', '<3 37D>'],
    5: ['<1 45E>', '<3 75G>', '<5 67H>'],
    6: ['<1 103F>', '<3 127B>', '<5 147H>', '<7 111A>', '<11 155E>'],
    7: [''],
    8: [''],
    9: [''],
    10: [''],
    11: [''],
    12: [''],
    13: [''],
    14: [''],
    15: [''],
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

    S_0 = S.copy()

    V = np.matmul(A % 2, S % 2) % 2
    S_Tnext = np.matmul(V, B % 2) % 2

    sequence.append(S[i - 1][j - 1])

    while not np.array_equal(S_0, S_Tnext):
        sequence.append(S_Tnext[i - 1][j - 1])
        V = np.matmul(A % 2, S_Tnext % 2) % 2
        S_Tnext = np.matmul(V, B % 2) % 2

    return sequence


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
    nodes = []
    res = []
    for state in state_matrix:
        states.remove(state)
        j += 1

        result = 0
        next_state = []
        for i, c in enumerate(polynom_coefficients):
            if c:
                result += state[i]
        next_state.append(result % 2)
        next_state.extend(state[:len(state) - 1])
        res.append((j, state,int(''.join(map(str, state)), 2)))

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
        if seq[i]:
            continue
        else:
            seq[i] = -1
    return np.array(seq)


def generate_acf_image(sequence):
    sequence = normalize_seq(sequence)
    acf = np.correlate(sequence, sequence, mode='same')

    acf_normalized = normalize_acf(acf, sequence)
    lags = np.arange(len(acf_normalized))

    plt.clf()

    # peaks, _ = find_peaks(acf_normalized)
    #
    # # Выбираем только наиболее высокие пики и впадины
    # peak_values = acf_normalized[peaks]
    # threshold = 0.9  # Устанавливаем порог в 95% от максимального значения
    # if len(peak_values):
    #     peaks = peaks[peak_values > np.percentile(peak_values, 100 * threshold)]
    # Строим график АКФ
    plt.plot(lags, acf_normalized, color='tab:green', linewidth=0.5)
    # Отмечаем пики красными треугольниками
    # plt.plot(peaks, acf_normalized[peaks], "r^", markersize=3)
    plt.grid(True)
    # # Подписываем значения y для каждого пика
    # for peak in peaks:
    #     plt.text(peak, acf_normalized[peak], f'{acf_normalized[peak]:.2f}', ha='right', va='bottom', fontsize=6)

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




def feedback_shift_generator(request):
    return render(request, 'feedback_shift_generator.html', context={'polynomes': polynomes_dict})

def matrix_shift_register(request):
    return render(request, 'matrix_shift_register.html', context={'polynomes': polynomes_dict})


def create_feedback_shift_generator(request):
    mode = request.GET.get('mode')
    seed_numbers = request.GET.getlist('seedNumbers[]', [])
    if mode == "2":
        polynom = request.GET.getlist('seedPolNumbers[]', [])
        polynom_coefficients = list(map(int, polynom))
    else:
        polynom = request.GET.get('valuesSelect')
        polynom_coefficients = get_polynomial(polynom)


    struct_matrix = create_struct_matrix_var_1(len(polynom_coefficients), polynom_coefficients)
    states = get_states(polynom_coefficients)
    seed_numbers = list(map(int, seed_numbers))
    state_matrix = [seed_numbers]

    results = {}

    while len(states):
        _, length, res = generate_states(states, state_matrix, polynom_coefficients)
        results[length] = res

    biggest_cycle = results[max(results.keys())]

    sequence = []
    for i in range(len(biggest_cycle)):
        sequence.append(biggest_cycle[i][1][0])

    hemming_weight = sum(1 for x in sequence if x == 1)

    t_exp_period = len(sequence)

    if mode == "2":
        t_period = t_exp_period

    else:
        j = 1
        match = re.search(r'<(\d+)', polynom)
        if match:
            j = int(match.group(1))

        t_period = calc_t(j, len(polynom_coefficients))

    image_base64 = generate_acf_image(sequence.copy())

    result_container_html = render_to_string(
        'generate_fsg.html', {
            'sequence': sequence,
            'hemming_weight': hemming_weight,
            'struct_matrix': struct_matrix,
            'biggest_cycle': biggest_cycle,
            't_period': t_period,
            't_exp_period': t_exp_period,
            'acf_image': image_base64
        }, request=request
    )
    response_data = {
        'message': 'Створено послідовність',
        'result_container_html': result_container_html,
    }

    return JsonResponse(response_data)


def create_matrix_shift_register(request):
    mode = request.GET.get('mode')
    if mode == "2":
        polynom_A = request.GET.getlist('seedPolAInputs[]', [])
        polynom_B = request.GET.getlist('seedPolBInputs[]', [])
        polynom_coefficients_A = list(map(int, polynom_A))
        polynom_coefficients_B = list(map(int, polynom_B))
    else:
        polynom_A = request.GET.get('valuesSelect_A')
        polynom_B = request.GET.get('valuesSelect_B')
        polynom_coefficients_A = get_polynomial(polynom_A)
        polynom_coefficients_B = get_polynomial(polynom_B)

    selected_rang = int(request.GET.get('selectedRang'))
    selected_i = int(request.GET.get('i'))
    selected_j = int(request.GET.get('j'))

    struct_matrix_A = create_struct_matrix_var_1(len(polynom_coefficients_A), polynom_coefficients_A)
    struct_matrix_B = create_struct_matrix_var_2(len(polynom_coefficients_B), polynom_coefficients_B)
    matrix_S = create_S_matrix(selected_rang, len(polynom_coefficients_A), len(polynom_coefficients_B))

    sequence = create_sequence(selected_i, selected_j, struct_matrix_A, struct_matrix_B, matrix_S)

    if mode == "2":
        t_period_A = 2 ** (len(polynom_coefficients_A)) - 1
        t_period_B = 2 ** (len(polynom_coefficients_B)) - 1
        t_period_C = len(sequence)

    else:
        j_1 = 1
        j_2 = 1
        match_A = re.search(r'<(\d+)', polynom_A)
        match_B = re.search(r'<(\d+)', polynom_B)
        if match_A:
            j_1 = int(match_A.group(1))
        if match_B:
            j_2 = int(match_B.group(1))

        t_period_A = int(calc_t(j_1, len(polynom_coefficients_A)) / gcd(calc_t(j_1, len(polynom_coefficients_A)), j_1))
        t_period_B = int(calc_t(j_2, len(polynom_coefficients_B)) / gcd(calc_t(j_2, len(polynom_coefficients_B)), j_2))
        t_period_C = int((t_period_A * t_period_B) / gcd(t_period_A, t_period_B))
    t_exp_period_C = len(sequence)

    hemming_weight = calculate_hemming_weight(selected_rang, len(polynom_coefficients_A), len(polynom_coefficients_B))
    hemming_exp_weight = sum(1 for x in sequence if x == 1)

    image_base64 = generate_acf_image(sequence.copy())


    result_container_html = render_to_string(
        'generate_msg.html', {
            'struct_matrix_A': struct_matrix_A,
            'struct_matrix_B': struct_matrix_B,
            'matrix_S': matrix_S,
            'sequence': sequence,
            't_period_A': t_period_A,
            't_period_B': t_period_B,
            't_period_C': t_period_C,
            't_exp_period_C': t_exp_period_C,
            'hemming_weight': hemming_weight,
            'hemming_exp_weight': hemming_exp_weight,
            'acf_image': image_base64,
        }, request=request
    )
    response_data = {
        'message': 'Створено послідовність',
        'result_container_html': result_container_html,
    }

    return JsonResponse(response_data)

def index(request):
    return render(request, 'main.html')
