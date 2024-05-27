import re
import scipy.signal

from django.http import JsonResponse
from django.shortcuts import render
from django.template.loader import render_to_string

from main.utils import get_polynomial, create_struct_matrix_var_1, get_states, generate_states, calc_t, factorize, \
    create_pvt_matrix, normalize_seq, generate_pvt_acf_image, generate_acf_image, create_struct_matrix_var_2, \
    create_S_matrix, create_sequence, calculate_hemming_weight, gcd, create_pvt_matrix_var_2, generate_torus, \
    generate_two_dim_acf_image, autocorrelation, polynomes_dict, generate_two_dim_acf_image_min, create_pvt_matrix1


def feedback_shift_generator(request):
    return render(request, 'feedback_shift_generator.html', context={'polynomes': polynomes_dict})


def matrix_shift_register(request):
    return render(request, 'matrix_shift_register.html', context={'polynomes': polynomes_dict})


def autocorr(request):
    return render(request, '2d_autocorr.html', context={'polynomes': polynomes_dict})


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

    _, seq_copy, n, m = factorize(sequence.copy())
    norm_sequence = normalize_seq(seq_copy)
    pvt_matrix = create_pvt_matrix_var_2(norm_sequence, n, m)

    acf_image_pvt = generate_pvt_acf_image(pvt_matrix)
    image_base64 = generate_acf_image(sequence.copy())

    result_container_html = render_to_string(
        'generate_fsg.html', {
            'sequence': sequence,
            'hemming_weight': hemming_weight,
            'struct_matrix': struct_matrix,
            'biggest_cycle': biggest_cycle,
            't_period': t_period,
            't_exp_period': t_exp_period,
            'acf_image': image_base64,
            'acf_image_pvt': acf_image_pvt,
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

    sequence, states = create_sequence(selected_i, selected_j, struct_matrix_A, struct_matrix_B, matrix_S)

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
            'states': states,
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


def create_autocorr(request):
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

    sequence, states = create_sequence(selected_i, selected_j, struct_matrix_A, struct_matrix_B, matrix_S)

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

    f, seq_copy, n, m = factorize(sequence.copy())
    norm_sequence = normalize_seq(seq_copy)

    pvt_matrix_var2 = create_pvt_matrix_var_2(norm_sequence, n, m)
    acf_image_pvt_var_2 = generate_pvt_acf_image(pvt_matrix_var2)

    if not f:
        pvt_matrix = create_pvt_matrix(norm_sequence, n, m)
        acf_image_pvt = generate_pvt_acf_image(pvt_matrix)
    else:
        acf_image_pvt = acf_image_pvt_var_2

    result_container_html = render_to_string(
        'generate_2d_autocorr.html', {
            'struct_matrix_A': struct_matrix_A,
            'struct_matrix_B': struct_matrix_B,
            'matrix_S': matrix_S,
            'sequence': sequence,
            'states': states,
            't_period_A': t_period_A,
            't_period_B': t_period_B,
            't_period_C': t_period_C,
            't_exp_period_C': t_exp_period_C,
            'hemming_weight': hemming_weight,
            'hemming_exp_weight': hemming_exp_weight,
            'acf_image_pvt': acf_image_pvt,
            'acf_image_pvt_var_2': acf_image_pvt_var_2,
        }, request=request
    )
    response_data = {
        'message': 'Створено послідовність',
        'result_container_html': result_container_html,
    }

    return JsonResponse(response_data)


def create_torus_autocorr(request):
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

    if mode == "2":
        t_period_A = 2 ** (len(polynom_coefficients_A)) - 1
        t_period_B = 2 ** (len(polynom_coefficients_B)) - 1

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

    torus = generate_torus(t_period_A, t_period_B, struct_matrix_A, struct_matrix_B, matrix_S)

    acf_image_torus = generate_two_dim_acf_image(t_period_A, t_period_B, matrix_S, torus, scipy.signal.correlate2d,
                                                 mode=0)
    acf_image_torus_a = generate_two_dim_acf_image(t_period_A, t_period_B, matrix_S, torus, autocorrelation, mode=1)
    sequence, states = create_sequence(selected_i, selected_j, struct_matrix_A, struct_matrix_B, matrix_S)
    acf_image_torus_min = generate_two_dim_acf_image_min(sequence, states, matrix_S)

    result_container_torus_html = render_to_string(
        'generate_2d_autocorr_torus.html', {
            'acf_image_torus': acf_image_torus,
            'acf_image_torus_a': acf_image_torus_a,
            'acf_image_torus_min': acf_image_torus_min,

        }, request=request
    )
    response_data = {
        'message': 'Створено послідовність',
        'result_container_torus_html': result_container_torus_html,
    }

    return JsonResponse(response_data)


def index(request):
    return render(request, 'main.html')
