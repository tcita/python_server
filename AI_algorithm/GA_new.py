import random
import numpy as np
import time
import pickle
from multiprocessing import Pool
import os  # Added for directory creation


# ---------------------------
# 游戏规则函数 (根据新规则调整和确认)
# ---------------------------

# 初始化一副牌（1到13，4种花色） - OK
def init_deck():
    """Initializes a standard 52-card deck (4 suits, values 1-13)."""
    return [i for i in range(1, 14) for _ in range(4)]


# 洗牌并发牌 - A玩家6张(唯一), B玩家3张 - OK (符合规则)
def deal_cards():
    """Deals 6 unique cards to A and 3 cards to B from a shuffled deck."""
    deck = init_deck()
    random.shuffle(deck)

    A = []
    a_unique_check = set()
    # Draw unique cards for A
    while len(A) < 6 and deck:  # Ensure deck has cards
        card = deck.pop()
        if card not in a_unique_check:
            a_unique_check.add(card)
            A.append(card)
        # If we can't find 6 unique cards (highly unlikely with 52 cards), something is wrong
        if not deck and len(A) < 6:
            print(f"Warning: Could only deal {len(A)} unique cards to A. Deck depleted early.")

    # Draw 3 cards for B from remaining deck
    B = []
    for _ in range(3):
        if deck:  # Check if deck is not empty
            B.append(deck.pop())
        else:
            print(f"Warning: Could only deal {len(B)} cards to B. Deck depleted.")
            break

    # print(f"Dealt A: {A}, B: {B}") # Debug print
    return A, B


# 模拟插入、匹配、得分、移除的核心逻辑 - 保留原实现，它符合描述中的匹配和得分规则
def simulate_insertion(A, x, pos):
    """
    Simulates inserting card x at position pos in A, calculates score, and returns results.
    Finds the nearest match (left or right), breaks distance ties with interval sum.
    Args:
        A (list): Player A's current hand.
        x (int): Card from B to insert.
        pos (int): Position index where x is inserted (0 to len(A)).

    Returns:
        tuple: (score, removal_length, new_length, match_found, new_A)
               score: Score obtained from this insertion (sum of removed interval).
               removal_length: Number of cards removed.
               new_length: Length of A after removal.
               match_found: 1 if a match led to removal, 0 otherwise.
               new_A: Player A's hand after insertion and potential removal.
    """
    candidate_A = A.copy()

    # Insert x at the specified position (0 to len(A))
    candidate_A.insert(pos, x)

    # Search for the nearest matching element y == x
    left_idx, right_idx = None, None
    min_left_dist, min_right_dist = float('inf'), float('inf')

    # Search left (indices < pos)
    for i in range(pos - 1, -1, -1):
        if candidate_A[i] == x:
            left_idx = i
            min_left_dist = pos - i
            break  # Found nearest left match

    # Search right (indices > pos)
    for j in range(pos + 1, len(candidate_A)):
        if candidate_A[j] == x:
            right_idx = j
            min_right_dist = j - pos
            break  # Found nearest right match

    # Determine which match to use (if any)
    chosen_match_idx = None
    if left_idx is not None and right_idx is not None:  # Matches on both sides
        if min_left_dist < min_right_dist:
            chosen_match_idx = left_idx
        elif min_right_dist < min_left_dist:
            chosen_match_idx = right_idx
        else:  # Equidistant matches, use interval sum tie-breaker
            left_interval = candidate_A[left_idx: pos + 1]
            right_interval = candidate_A[pos: right_idx + 1]
            if sum(left_interval) >= sum(right_interval):  # Prioritize left if sums are equal too
                chosen_match_idx = left_idx
            else:
                chosen_match_idx = right_idx
    elif left_idx is not None:
        chosen_match_idx = left_idx
    elif right_idx is not None:
        chosen_match_idx = right_idx
    else:
        # No match found
        return 0, 0, len(candidate_A), 0, candidate_A

    # Calculate score and remove interval based on the chosen match
    if chosen_match_idx < pos:  # Matched left
        start, end = chosen_match_idx, pos
    else:  # Matched right
        start, end = pos, chosen_match_idx

    removal_interval = candidate_A[start: end + 1]
    score = sum(removal_interval)
    removal_length = len(removal_interval)

    # Create the new A by removing the interval
    new_A = candidate_A[:start] + candidate_A[end + 1:]
    new_length = len(new_A)
    match_found = 1

    return score, removal_length, new_length, match_found, new_A


# ---------------------------
# 遗传算法相关函数 (根据新规则调整)
# ---------------------------

# REMOVED: get_best_insertion_score, get_updated_A_after_insertion (no longer needed)
# REMOVED: calculate_future_score (no longer applicable due to random B order)

# 基于基因组选择插入位置 (已移除 future_score)
def genome_choose_insertion(genome, A, x):
    """
    Chooses the best insertion position for card x into hand A based on the genome evaluation.
    The genome is now length 4, evaluating [score, removal_length, new_length, match_found].

    Args:
        genome (list): The length-4 genome (weights).
        A (list): Player A's current hand.
        x (int): The card from B to be inserted.

    Returns:
        tuple: (best_pos, score, new_A)
               best_pos: The chosen insertion position index.
               score: The score obtained from inserting at best_pos.
               new_A: Player A's hand after inserting at best_pos and potential removal.
    """
    best_value = -float('inf')
    best_move_info = None  # To store (pos, score, new_A) for the best move

    possible_moves = []  # Store results for all potential positions

    # Iterate through all len(A) + 1 possible insertion positions
    for pos in range(len(A) + 1):
        # Simulate the insertion for this position
        score, removal_length, new_length, match_found, resultant_A = simulate_insertion(A, x, pos)

        # Create the feature vector (Length 4 NOW)
        features = np.array([score, removal_length, new_length, match_found], dtype=float)

        # Calculate the value using the genome
        value = np.dot(genome, features)

        # Store the outcome for this position
        possible_moves.append({'value': value, 'pos': pos, 'score': score, 'new_A': resultant_A})

    # Handle the case where A might be empty initially or becomes empty
    if not possible_moves:
        # This should only happen if A was empty. The loop range(0, 1) runs once for pos=0.
        # simulate_insertion handles inserting into empty A correctly.
        # Let's ensure we handle the very first insertion gracefully if A starts empty.
        if not A:
            score, _, _, _, resultant_A = simulate_insertion(A, x, 0)
            # Return position 0, score, and the new A (which just contains x)
            return 0, score, resultant_A
        else:
            # This state should theoretically not be reached if A is not empty.
            # If it were, it implies no valid positions, which contradicts range(len(A)+1).
            # As a fallback, maybe just insert at the end with 0 score? Or raise error?
            # Let's trust simulate_insertion and the loop logic. If possible_moves is empty
            # when A is not, something fundamental is wrong.
            # For robustness, let's default to inserting at end if this unexpected case occurs.
            print(f"Warning: possible_moves list empty unexpectedly for A={A}, x={x}. Defaulting.")
            new_A = A.copy()
            new_A.append(x)
            return len(A), 0, new_A

    # Choose the move with the highest calculated 'value'
    best_move = max(possible_moves, key=lambda move: move['value'])

    # Return the best position, the actual score from that move, and the resulting A
    return best_move['pos'], best_move['score'], best_move['new_A']


# 模拟一轮完整游戏 (B的顺序随机化)
def simulate_round(genome):
    """Simulates a single round of the game with shuffled B order."""
    A, B = deal_cards()
    if not B:  # Handle edge case where B couldn't be fully dealt
        return 0

    round_score = 0
    # IMPORTANT: Shuffle B's order for this specific round simulation
    shuffled_B = random.sample(B, len(B))

    current_A = A.copy()  # Use a copy of A that updates within the round

    for x in shuffled_B:
        # Genome chooses insertion based on current A and card x
        # Note: We don't need lookahead (remaining_B) anymore
        pos, score, current_A = genome_choose_insertion(genome, current_A, x)
        round_score += score
        # current_A is implicitly updated for the next iteration

    return round_score


# 评估基因组适应度 - OK (using mean/median combo)
def evaluate_genome(genome, num_rounds=1000):
    """Evaluates a genome's fitness by simulating many rounds."""
    # Ensure genome has the correct length (4)
    if len(genome) != 4:
        print(f"Error: Genome length is {len(genome)}, expected 4. Returning low fitness.")
        return -float('inf')  # Penalize incorrect genomes

    scores = [simulate_round(genome) for _ in range(num_rounds)]
    if not scores: return 0  # Handle case where no rounds could be played

    # Using median and mean might be sensitive to outliers if scores vary wildly.
    # Consider adjusting weights or using only mean/median if results seem unstable.
    median_score = np.median(scores)
    mean_score = np.mean(scores)

    # Fitness combines robustness (median) and average performance (mean)
    fitness = 0.7 * median_score + 0.3 * mean_score
    return fitness


# 使用多进程评估 - OK
def evaluate_genomes_with_processes(population, num_rounds=1000, num_processes=8):
    """Evaluates a population of genomes in parallel."""
    # Make sure num_processes is reasonable
    num_processes = min(num_processes, os.cpu_count())
    try:
        with Pool(processes=num_processes) as pool:
            # Create argument tuples for starmap
            tasks = [(genome, num_rounds) for genome in population]
            fitnesses = pool.starmap(evaluate_genome, tasks)
        return fitnesses
    except Exception as e:
        print(f"Error during multiprocessing evaluation: {e}")
        # Fallback to serial evaluation if pooling fails
        print("Falling back to serial evaluation...")
        return [evaluate_genome(genome, num_rounds) for genome in population]


# 遗传算法主过程 (使用长度为4的基因组)
def genetic_algorithm(pop_size=100, generations=50, num_rounds=500,  # Adjusted defaults for potentially faster runs
                      elitism_ratio=0.1, tournament_size=3, mutation_rate_initial=0.3,
                      mutation_rate_final=0.05, mutation_strength=0.5, num_processes=8):
    """
    Runs the genetic algorithm to find the best genome (length 4).
    """
    start_time = time.time()

    # Initialize population with length-4 genomes
    population = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(pop_size)]

    best_fitness_history, avg_fitness_history = [], []
    best_genome_overall, best_fitness_overall = None, -float('inf')

    elitism_count = int(elitism_ratio * pop_size)

    print(f"Starting GA: Pop Size={pop_size}, Generations={generations}, Rounds/Eval={num_rounds}, Features=4")

    for gen in range(generations):
        gen_start_time = time.time()

        # Evaluate fitness of the current population
        fitnesses = evaluate_genomes_with_processes(population, num_rounds, num_processes)

        gen_best_fitness = max(fitnesses)
        gen_avg_fitness = np.mean(fitnesses)
        best_idx_gen = fitnesses.index(gen_best_fitness)

        best_fitness_history.append(gen_best_fitness)
        avg_fitness_history.append(gen_avg_fitness)

        # Update overall best if current generation is better
        if gen_best_fitness > best_fitness_overall:
            best_fitness_overall = gen_best_fitness
            best_genome_overall = population[best_idx_gen][:]  # Store a copy
            print(
                f"*** New Best Overall Fitness: {best_fitness_overall:.4f} (Genome: {np.round(best_genome_overall, 4)}) ***")

        gen_time = time.time() - gen_start_time
        print(
            f"Generation {gen + 1}/{generations}: Best Fitness = {gen_best_fitness:.4f}, Avg Fitness = {gen_avg_fitness:.4f}, Time: {gen_time:.2f}s")

        # --- Selection ---
        # Sort population by fitness (descending)
        sorted_indices = sorted(range(pop_size), key=lambda k: fitnesses[k], reverse=True)
        sorted_population = [population[i] for i in sorted_indices]

        # Elitism: Carry over the top individuals
        elites = sorted_population[:elitism_count]

        # Tournament Selection for the rest
        selected_for_breeding = []
        num_to_select = pop_size - elitism_count
        for _ in range(num_to_select):
            # Select participants for the tournament
            tournament_indices = random.sample(range(pop_size), tournament_size)
            # Find the winner (highest fitness) within the tournament
            winner_idx = max(tournament_indices, key=lambda i: fitnesses[i])
            selected_for_breeding.append(population[winner_idx])

        # --- Reproduction ---
        next_population = elites[:]  # Start next generation with elites

        # Crossover and Mutation
        while len(next_population) < pop_size:
            parent1, parent2 = random.sample(selected_for_breeding, 2)

            # Crossover (Average) - Keep it simple for now
            crossover_prob = 0.7
            child = []
            for p1_gene, p2_gene in zip(parent1, parent2):
                if random.random() < crossover_prob:
                    # Blend genes (average)
                    child.append((p1_gene + p2_gene) / 2)
                else:
                    # Inherit from one parent (e.g., parent1)
                    child.append(p1_gene)

            # Mutation (Gaussian, with decaying rate)
            current_mutation_rate = max(mutation_rate_final, mutation_rate_initial - (gen / generations) * (
                        mutation_rate_initial - mutation_rate_final))
            mutated_child = []
            for gene in child:
                if random.random() < current_mutation_rate:
                    mutation = random.gauss(0, mutation_strength)
                    mutated_child.append(gene + mutation)
                else:
                    mutated_child.append(gene)

            # Optional: Clamp gene values if they stray too far (e.g., [-2, 2])
            # mutated_child = [max(-2.0, min(2.0, gene)) for gene in mutated_child]

            next_population.append(mutated_child)

        population = next_population  # Update population for the next generation

    end_time = time.time()
    print(f"\nGA Finished. Total Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Best Genome Found: {np.round(best_genome_overall, 5)}")
    print(f"Best Fitness Achieved: {best_fitness_overall:.4f}")

    return best_genome_overall


# ---------------------------
# Utility and Main Execution
# ---------------------------

def save_best_genome(best_genome, filename="trained/best_genome_v2.pkl"):
    """Saves the best genome to a pickle file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, 'wb') as file:
            pickle.dump(best_genome, file)
        print(f"Best genome saved to {filename}")
    except Exception as e:
        print(f"Error saving genome to {filename}: {e}")


def load_best_genome(filename="trained/best_genome_v2.pkl"):
    """Loads a genome from a pickle file."""
    try:
        with open(filename, 'rb') as file:
            genome = pickle.load(file)
        print(f"Genome loaded from {filename}")
        # Add a check for genome length
        if len(genome) != 4:
            print(f"Warning: Loaded genome has length {len(genome)}, expected 4.")
        return genome
    except FileNotFoundError:
        print(f"Error: Genome file not found at {filename}")
        return None
    except Exception as e:
        print(f"Error loading genome from {filename}: {e}")
        return None


# 展示策略在一个给定牌局上的执行过程 (使用随机B顺序)
# (确保其他函数如 init_deck, deal_cards, simulate_insertion, genome_choose_insertion,
#  genetic_algorithm, save_best_genome, load_best_genome 等保持不变，
#  特别是 genome_choose_insertion 确认是接收长度为4的genome且不依赖remaining_B)

def Get_GA_Strategy(best_genome, A_orig, B_orig):
    """
    Demonstrates the strategy for a given hand A and B, using one random B order.
    Outputs the strategy as a list of (original_B_index, chosen_pos) tuples,
    ordered by the random processing sequence.

    Args:
        best_genome (list): The trained (length-4) genome.
        A_orig (list): Initial hand A.
        B_orig (list): Initial hand B.

    Returns:
        dict: Contains the strategy log (list of tuples) and total score for this run.
              Returns None if inputs are invalid.
    """
    if not best_genome or len(best_genome) != 4:
        print("Error: Invalid or missing genome.")
        return None

    if not B_orig:
        print("Initial B is empty, no moves to demonstrate.")
        return {"strategy_log": [], "total_score": 0}

    A = A_orig.copy()
    B = B_orig.copy() # Keep a copy if needed, though B_orig is used mainly

    # 1. Create list of (original_index, card_value) pairs from B_orig
    indexed_B = list(enumerate(B_orig)) # e.g., [(0, 9), (1, 5), (2, 2)] for B=[9, 5, 2]

    # 2. Shuffle this list of pairs to get the random processing order
    shuffled_indexed_B = random.sample(indexed_B, len(indexed_B))
    # e.g., could be [(2, 2), (0, 9), (1, 5)]

    round_score = 0
    num_moves = 0
    strategy = []  # To store (original_B_index, chosen_pos) tuples
    detailed_log = [] # Optional: For more verbose logging


    processing_order_cards = [card for idx, card in shuffled_indexed_B]


    current_A = A # Use a copy that updates through the process

    # 3. Iterate through the shuffled pairs
    for move_num, (original_index, card_value) in enumerate(shuffled_indexed_B):


        # Genome chooses the best insertion position based on current A and the card's value
        pos, score, current_A = genome_choose_insertion(best_genome, current_A, card_value)



        # 4. Append (original_index, pos) to the strategy list
        strategy.append((original_index, pos))



        round_score += score
        num_moves += 1



    # Return both the simple strategy list and the total score
    return strategy

# --- Example Usage (within if __name__ == "__main__": block) ---
# (Assuming best_genome is loaded or trained)
# if loaded_genome:
#     print("\nRunning demonstration with a sample deal...")
#     test_A, test_B = deal_cards()
#     while not test_B:
#          print("B was empty, dealing again for demo...")
#          test_A, test_B = deal_cards()
#
#     # Run the corrected demonstration function
#     demo_result = Get_GA_Strategy(loaded_genome, test_A, test_B)
#     if demo_result:
#         # You can access the strategy list like this:
#         final_strategy_list = demo_result['strategy_log']
#         print(f"\nExtracted Strategy List for potential use: {final_strategy_list}")
if __name__ == "__main__":
    # Set GA parameters (adjust as needed)
    POP_SIZE = 200  # Increased population size for more exploration
    GENERATIONS = 80  # More generations for convergence
    NUM_ROUNDS_EVAL = 1000  # Higher rounds for more stable fitness evaluation
    NUM_PROCESSES = os.cpu_count()  # Use available CPU cores

    # --- Train the Model ---
    print("Starting Genetic Algorithm Training...")
    best_genome = genetic_algorithm(
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        num_rounds=NUM_ROUNDS_EVAL,
        num_processes=NUM_PROCESSES,
        # Other parameters like elitism, mutation can also be tuned here
    )

    # --- Save the Best Genome ---
    if best_genome:
        save_best_genome(best_genome, filename="trained/best_game_genome_len4.pkl")  # New filename

        # --- Load and Demonstrate ---
        print("\nLoading the best genome for demonstration...")
        loaded_genome = load_best_genome(filename="trained/best_game_genome_len4.pkl")

        if loaded_genome:
            print("Running demonstration with a sample deal...")
            # Get a fresh deal for demonstration
            test_A, test_B = deal_cards()
            # Ensure B is not empty for a meaningful demo
            while not test_B:
                print("B was empty, dealing again for demo...")
                test_A, test_B = deal_cards()

            Get_GA_Strategy(loaded_genome, test_A, test_B)
        else:
            print("Could not load genome for demonstration.")
    else:
        print("Genetic algorithm did not produce a best genome.")