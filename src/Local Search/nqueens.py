"""
=============================================================================
Local Search: N-Queens Problem -- Modul Logika Utama
=============================================================================
Modul ini menyediakan seluruh fungsi inti untuk persoalan N-Queens:

Representasi & Evaluasi:
  - generate_random_state(n)   : Membangkitkan state random
  - calculate_h(state)         : Objective function (jumlah attacking pairs)
  - get_neighbors(state)       : Successor function (seluruh neighbor)

Algoritma Local Search:
  - hill_climbing_basic(n)             : Steepest-Ascent Hill-Climbing
  - hill_climbing_sideways(n, M)       : HC dengan Sideways Move
  - hill_climbing_stochastic(n)        : Stochastic Hill-Climbing
  - hill_climbing_random_restart(n, R) : Random Restart Hill-Climbing
  - simulated_annealing(n, T0, a, Tm)  : Simulated Annealing
  - genetic_algorithm(n, P, pm, G)     : Genetic Algorithm

Visualisasi Teks:
  - print_board(state)                 : Cetak papan N-Queens
  - print_state_array(state)           : Cetak array state
  - print_h_history(h_history)         : Cetak riwayat h
  - visualize_search(states, h_vals)   : Visualisasi proses pencarian
=============================================================================
"""

import random
import math

# ============================================================================
# REPRESENTASI STATE & FUNGSI DASAR
# ============================================================================

def generate_random_state(n):
    """
    Membangkitkan initial state secara random untuk papan n x n.
    Setiap elemen q_i in {1, ..., n} menyatakan posisi baris ratu pada kolom ke-i.
    """
    return [random.randint(1, n) for _ in range(n)]


def calculate_h(state):
    """
    Menghitung objective function h(S) = jumlah pasangan ratu yang saling menyerang.
    Dua ratu pada kolom i dan j saling menyerang jika:
      - Konflik baris:    q_i == q_j
      - Konflik diagonal: |q_i - q_j| == |i - j|
    Nilai h = 0 berarti solusi ditemukan (goal state).
    Kompatibel dengan encoding Windows (cp1252).
    """
    n = len(state)
    h = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Konflik baris
            if state[i] == state[j]:
                h += 1
            # Konflik diagonal
            elif abs(state[i] - state[j]) == abs(i - j):
                h += 1
    return h


def get_neighbors(state):
    """
    Membangkitkan seluruh neighbor dari state saat ini.
    Move: mengubah posisi baris satu ratu pada satu kolom.
    Total neighbor = N x (N - 1).
    Mengembalikan list of (state_baru, kolom_diubah, baris_baru).
    """
    n = len(state)
    neighbors = []
    for col in range(n):
        for row in range(1, n + 1):
            if row != state[col]:
                new_state = state[:]
                new_state[col] = row
                neighbors.append((new_state, col, row))
    return neighbors


# ============================================================================
# VISUALISASI TEKS
# ============================================================================

def print_board(state, label="", h_val=None):
    """
    Mencetak visualisasi papan N-Queens dalam format teks.
    Q = ratu, . = kosong
    """
    n = len(state)
    header = label
    if h_val is not None:
        header += f" | h = {h_val}"
    if header:
        print(header)

    print("  +" + "---+" * n)
    for row in range(1, n + 1):
        line = f"{row:2d}|"
        for col in range(n):
            if state[col] == row:
                line += " Q |"
            else:
                line += " . |"
        print(line)
        print("  +" + "---+" * n)
    # Label kolom
    col_label = "   "
    for col in range(1, n + 1):
        col_label += f" {col}  "
    print(col_label)
    print()


def print_state_array(state, label=""):
    """Mencetak representasi array dari state."""
    if label:
        print(f"{label}: {state}")
    else:
        print(f"State: {state}")


def print_separator(char="=", length=70):
    """Mencetak garis pemisah."""
    print(char * length)


def print_h_history(h_history, label="Riwayat h"):
    """Mencetak riwayat nilai objective function."""
    print(f"\n{label}:")
    print(f"{'Iterasi':>10} | {'h(S)':>6}")
    print("-" * 20)
    for i, h in enumerate(h_history):
        print(f"{i:>10} | {h:>6}")


def visualize_search(states, h_values, algorithm_name, step_label="Iterasi",
                     max_display=10):
    """
    Menampilkan visualisasi proses pencarian:
    - Perubahan state antar iterasi
    - Nilai h pada setiap iterasi
    Hanya menampilkan max_display state pertama dan terakhir jika terlalu banyak.
    """
    total = len(states)
    print(f"\n{'='*70}")
    print(f"  VISUALISASI PROSES PENCARIAN: {algorithm_name}")
    print(f"{'='*70}")
    print(f"  Total {step_label.lower()}: {total - 1}")
    print()

    if total <= max_display:
        # Tampilkan semua
        indices = list(range(total))
    else:
        # Tampilkan awal, tengah, dan akhir
        half = max_display // 2
        indices = list(range(half))
        indices.append(-1)  # Penanda "..."
        indices.extend(range(total - half, total))

    for idx in indices:
        if idx == -1:
            print(f"  ... ({total - max_display} iterasi lainnya tidak ditampilkan) ...\n")
            continue
        label_text = f"{step_label} {idx}"
        if idx == 0:
            label_text += " (STATE AWAL)"
        elif idx == total - 1:
            label_text += " (STATE AKHIR)"
        print_board(states[idx], label=label_text, h_val=h_values[idx])


# ============================================================================
# HILL-CLIMBING: BASIC (STEEPEST-ASCENT)
# ============================================================================

def hill_climbing_basic(n):
    """
    Basic Hill-Climbing (Steepest-Ascent) untuk N-Queens.
    Pada setiap iterasi, pilih neighbor dengan h terkecil.
    Berhenti jika tidak ada neighbor yang lebih baik.
    Mengembalikan: (state_akhir, h_akhir, h_history, states_history)
    """
    state = generate_random_state(n)
    h = calculate_h(state)

    h_history = [h]
    states_history = [state[:]]

    iteration = 0
    while True:
        neighbors = get_neighbors(state)

        # Cari neighbor dengan h terkecil
        best_neighbor = None
        best_h = h
        for neighbor_state, col, row in neighbors:
            neighbor_h = calculate_h(neighbor_state)
            if neighbor_h < best_h:
                best_h = neighbor_h
                best_neighbor = neighbor_state

        # Jika tidak ada perbaikan, berhenti
        if best_neighbor is None:
            break

        state = best_neighbor
        h = best_h
        iteration += 1
        h_history.append(h)
        states_history.append(state[:])

        if h == 0:
            break

    return state, h, h_history, states_history


# ============================================================================
# HILL-CLIMBING: SIDEWAYS MOVE
# ============================================================================

def hill_climbing_sideways(n, max_sideways=100):
    """
    Hill-Climbing dengan Sideways Move untuk N-Queens.
    Mengizinkan perpindahan ke neighbor dengan h sama (sideways move),
    dibatasi max_sideways langkah berturut-turut.
    Mengembalikan: (state_akhir, h_akhir, h_history, states_history)
    """
    state = generate_random_state(n)
    h = calculate_h(state)

    h_history = [h]
    states_history = [state[:]]
    sideways_count = 0

    while True:
        neighbors = get_neighbors(state)

        # Cari neighbor dengan h terkecil
        best_h = float('inf')
        best_neighbors = []
        for neighbor_state, col, row in neighbors:
            neighbor_h = calculate_h(neighbor_state)
            if neighbor_h < best_h:
                best_h = neighbor_h
                best_neighbors = [neighbor_state]
            elif neighbor_h == best_h:
                best_neighbors.append(neighbor_state)

        if best_h < h:
            # Perbaikan ditemukan
            state = random.choice(best_neighbors)
            h = best_h
            sideways_count = 0
        elif best_h == h and sideways_count < max_sideways:
            # Sideways move
            state = random.choice(best_neighbors)
            sideways_count += 1
        else:
            # Tidak ada perbaikan dan batas sideways tercapai
            break

        h_history.append(h)
        states_history.append(state[:])

        if h == 0:
            break

    return state, h, h_history, states_history


# ============================================================================
# STOCHASTIC HILL-CLIMBING
# ============================================================================

def hill_climbing_stochastic(n, max_iter=10000):
    """
    Stochastic Hill-Climbing untuk N-Queens.
    Memilih secara acak di antara neighbor yang memberikan perbaikan.
    Mengembalikan: (state_akhir, h_akhir, h_history, states_history)
    """
    state = generate_random_state(n)
    h = calculate_h(state)

    h_history = [h]
    states_history = [state[:]]

    for _ in range(max_iter):
        neighbors = get_neighbors(state)

        # Kumpulkan neighbor yang lebih baik
        better_neighbors = []
        for neighbor_state, col, row in neighbors:
            neighbor_h = calculate_h(neighbor_state)
            if neighbor_h < h:
                better_neighbors.append((neighbor_state, neighbor_h))

        if not better_neighbors:
            # Tidak ada neighbor yang lebih baik → berhenti
            break

        # Pilih secara acak dari neighbor yang lebih baik
        chosen_state, chosen_h = random.choice(better_neighbors)
        state = chosen_state
        h = chosen_h

        h_history.append(h)
        states_history.append(state[:])

        if h == 0:
            break

    return state, h, h_history, states_history


# ============================================================================
# RANDOM RESTART HILL-CLIMBING
# ============================================================================

def hill_climbing_random_restart(n, max_restarts=100):
    """
    Random Restart Hill-Climbing untuk N-Queens.
    Menjalankan basic hill-climbing berkali-kali dari initial state random baru.
    Mengembalikan: (state_akhir, h_akhir, h_history, states_history,
                    total_restarts, restart_points)
    """
    all_h_history = []
    all_states_history = []
    restart_points = []

    for restart in range(max_restarts):
        state, h, h_history, states_history = hill_climbing_basic(n)

        # Catat titik restart
        restart_points.append(len(all_h_history))
        all_h_history.extend(h_history)
        all_states_history.extend(states_history)

        if h == 0:
            # Solusi ditemukan
            return (state, h, all_h_history, all_states_history,
                    restart + 1, restart_points)

    # Gagal menemukan solusi setelah max_restarts
    return (state, h, all_h_history, all_states_history,
            max_restarts, restart_points)


# ============================================================================
# SIMULATED ANNEALING
# ============================================================================

def simulated_annealing(n, t0=100.0, alpha=0.995, t_min=0.01):
    """
    Simulated Annealing untuk N-Queens.
    - t0: suhu awal
    - alpha: cooling rate (eksponensial)
    - t_min: suhu minimum (kriteria berhenti)
    Mengembalikan: (state_akhir, h_akhir, h_history, states_history, temp_history)
    """
    state = generate_random_state(n)
    h = calculate_h(state)
    t = t0

    h_history = [h]
    states_history = [state[:]]
    temp_history = [t]

    while t > t_min and h != 0:
        # Pilih satu neighbor secara acak
        col = random.randint(0, n - 1)
        new_row = random.randint(1, n)
        while new_row == state[col]:
            new_row = random.randint(1, n)

        new_state = state[:]
        new_state[col] = new_row
        new_h = calculate_h(new_state)

        delta_e = new_h - h

        if delta_e < 0:
            # Neighbor lebih baik → selalu terima
            state = new_state
            h = new_h
        else:
            # Neighbor lebih buruk → terima dengan probabilitas e^(-ΔE/T)
            probability = math.exp(-delta_e / t)
            if random.random() < probability:
                state = new_state
                h = new_h

        t *= alpha  # Cooling

        h_history.append(h)
        states_history.append(state[:])
        temp_history.append(t)

    return state, h, h_history, states_history, temp_history


# ============================================================================
# GENETIC ALGORITHM
# ============================================================================

def ga_fitness(state):
    """
    Fitness function: fitness = h_max - h(S).
    h_max = C(N,2) = N*(N-1)/2 (jumlah pasangan maksimum).
    Semakin tinggi fitness, semakin baik.
    """
    n = len(state)
    h_max = n * (n - 1) // 2
    return h_max - calculate_h(state)


def ga_selection_roulette(population, fitnesses):
    """
    Roulette Wheel Selection.
    Probabilitas terpilih proporsional terhadap fitness.
    """
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choice(population)

    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitnesses):
        current += fitness
        if current >= pick:
            return individual[:]
    return population[-1][:]


def ga_crossover(parent1, parent2):
    """
    Single-Point Crossover.
    Pilih titik potong c secara acak, offspring = parent1[:c] + parent2[c:]
    """
    n = len(parent1)
    c = random.randint(1, n - 1)
    offspring = parent1[:c] + parent2[c:]
    return offspring


def ga_mutate(individual, mutation_rate=0.1):
    """
    Mutation: dengan probabilitas mutation_rate, pilih satu kolom secara acak
    dan ganti nilainya dengan nilai random baru.
    """
    if random.random() < mutation_rate:
        n = len(individual)
        col = random.randint(0, n - 1)
        individual[col] = random.randint(1, n)
    return individual


def genetic_algorithm(n, pop_size=100, mutation_rate=0.1, max_generations=500):
    """
    Genetic Algorithm untuk N-Queens.
    Mengembalikan: (individu_terbaik, h_terbaik, best_fitness_history,
                    avg_fitness_history, generation_found)
    """
    h_max = n * (n - 1) // 2

    # Inisialisasi populasi
    population = [generate_random_state(n) for _ in range(pop_size)]

    best_fitness_history = []
    avg_fitness_history = []
    best_individual_history = []

    for gen in range(max_generations):
        fitnesses = [ga_fitness(ind) for ind in population]
        best_fit = max(fitnesses)
        avg_fit = sum(fitnesses) / len(fitnesses)
        best_idx = fitnesses.index(best_fit)

        best_fitness_history.append(best_fit)
        avg_fitness_history.append(avg_fit)
        best_individual_history.append(population[best_idx][:])

        # Cek apakah solusi ditemukan (fitness = h_max berarti h = 0)
        if best_fit == h_max:
            return (population[best_idx], 0, best_fitness_history,
                    avg_fitness_history, gen + 1, best_individual_history)

        # Buat populasi baru
        new_population = []
        while len(new_population) < pop_size:
            parent1 = ga_selection_roulette(population, fitnesses)
            parent2 = ga_selection_roulette(population, fitnesses)
            offspring = ga_crossover(parent1, parent2)
            offspring = ga_mutate(offspring, mutation_rate)
            new_population.append(offspring)

        population = new_population

    # Kembalikan individu terbaik terakhir
    fitnesses = [ga_fitness(ind) for ind in population]
    best_idx = fitnesses.index(max(fitnesses))
    best_ind = population[best_idx]
    return (best_ind, calculate_h(best_ind), best_fitness_history,
            avg_fitness_history, max_generations, best_individual_history)



