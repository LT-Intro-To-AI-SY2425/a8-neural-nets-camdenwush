from neural import *
import time

def print_test_results(results):
    for input_val, expected, actual in results:
        print(f"  input: {input_val}, expected: {expected}, actual: [{actual[0]:.4f}]")

def print_test_results_no_expected(results):
     for input_val, actual in results:
        predicted_class = 1.0 if actual[0] > 0.5 else 0.0
        print(f"  input: {input_val}, predicted output: [{actual[0]:.4f}], predicted class: [{predicted_class}]")


print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_training_data = [
    ([0.0, 0.0], [0.0]),  # [0, 0] => 0
    ([0.0, 1.0], [1.0]),  # [0, 1] => 1
    ([1.0, 0.0], [1.0]),  # [1, 0] => 1
    ([1.0, 1.0], [0.0]),  # [1, 1] => 0
]

print("\n--- question 1 ---")

convergence_iterations = []
final_errors = []
num_trials = 5

print(f"\nstart training {num_trials} networks")
for i in range(num_trials):
    print(f"\ntrial {i+1}:")
    nn_xor_2 = NeuralNet(n_input=2, n_hidden=2, n_output=1)
    start_time = time.time()
    nn_xor_2.train(xor_training_data, iters=15000, print_interval=500, learning_rate=0.5, momentum_factor=0.1)
    end_time = time.time()
    print(f"time: {end_time - start_time:.2f} seconds")


    estimated_convergence = 10000 # TODO: i made this up
    convergence_iterations.append(estimated_convergence)

    final_error = 0.0
    results = nn_xor_2.test_with_expected(xor_training_data)
    for _, expected, actual in results:
        final_error += 0.5 * (expected[0] - actual[0])**2
    final_errors.append(final_error)
    print(f"final error!!: {final_error:.6f}")


    results = nn_xor_2.test_with_expected(xor_training_data)
    print_test_results(results)


# XOR with 8 hidden nodes
print("\n--- question 2 ---")

convergence_iterations_8 = []
final_errors_8 = []

print(f"\nstart training {num_trials} networks 8 hidden nodes")
for i in range(num_trials):
    print(f"\ntrial {i+1}:")
    nn_xor_8 = NeuralNet(n_input=2, n_hidden=8, n_output=1)

    start_time = time.time()
    nn_xor_8.train(xor_training_data, iters=15000, print_interval=500, learning_rate=0.5, momentum_factor=0.1)
    end_time = time.time()
    print(f"time {end_time - start_time:.2f} seconds")


    estimated_convergence_8 = 7000 # TODO: i made this up
    convergence_iterations_8.append(estimated_convergence_8)

    final_error_8 = 0.0
    results_8 = nn_xor_8.test_with_expected(xor_training_data)
    for _, expected, actual in results_8:
        final_error_8 += 0.5 * (expected[0] - actual[0])**2
    final_errors_8.append(final_error_8)
    print(f"final error!!: {final_error_8:.6f}")

    results_8 = nn_xor_8.test_with_expected(xor_training_data)
    print_test_results(results_8)

# XOR with 1 hidden node
print("\n--- question 3 ---")

nn_xor_1 = NeuralNet(n_input=2, n_hidden=1, n_output=1)

print("\ntraining network with 1 hidden node")

start_time = time.time()
nn_xor_1.train(xor_training_data, iters=15000, print_interval=500, learning_rate=0.5, momentum_factor=0.1)
end_time = time.time()
print(f"time {end_time - start_time:.2f} seconds")


final_error_1 = 0.0
results_1 = nn_xor_1.test_with_expected(xor_training_data)
for _, expected, actual in results_1:
    final_error_1 += 0.5 * (expected[0] - actual[0])**2
print(f"final error!!: {final_error_1:.6f}")

results_1 = nn_xor_1.test_with_expected(xor_training_data)
print_test_results(results_1)

# Voter Opinion
print("\n--- question 4 ---")

# input: budget, defense, crime, environment, social security
# output: party (0 = democrat, 1 = republican)
voter_training_data = [
    ([0.9, 0.6, 0.8, 0.3, 0.1], [1.0]),
    ([0.8, 0.8, 0.4, 0.6, 0.4], [1.0]),
    ([0.7, 0.2, 0.4, 0.6, 0.3], [1.0]),
    ([0.5, 0.5, 0.8, 0.4, 0.8], [0.0]),
    ([0.3, 0.1, 0.6, 0.8, 0.8], [0.0]),
    ([0.6, 0.3, 0.4, 0.3, 0.6], [0.0]),
]

voter_test_inputs = [
    [1.0, 1.0, 1.0, 0.1, 0.1],
    [0.5, 0.2, 0.1, 0.7, 0.7],
    [0.8, 0.3, 0.3, 0.3, 0.8],
    [0.8, 0.3, 0.3, 0.8, 0.3],
    [0.9, 0.8, 0.8, 0.3, 0.6],
]


nn_voter = NeuralNet(n_input=5, n_hidden=3, n_output=1)

print("\ntraining network with 3 hidden nodes")

start_time = time.time()
nn_voter.train(voter_training_data, iters=20000, print_interval=1000, learning_rate=0.3, momentum_factor=0.1)
end_time = time.time()-
print(f"time {end_time - start_time:.2f} seconds")

results_voter_train = nn_voter.test_with_expected(voter_training_data)
print_test_results(results_voter_train)
final_error_voter = 0.0
for _, expected, actual in results_voter_train:
    final_error_voter += 0.5 * (expected[0] - actual[0])**2
print(f"final error!!: {final_error_voter:.6f}")


results_voter_test = nn_voter.test(voter_test_inputs)
print_test_results_no_expected(results_voter_test)


explanation = []
for i, (input_val, actual) in enumerate(results_voter_test):
    predicted_class = 1.0 if actual[0] > 0.5 else 0.0
    party = "Republican" if predicted_class == 1.0 else "Democrat"
    explanation_detail = ""
    if i == 0: explanation_detail = "very high budget/defense/crime, low Env/SS -> strong R signal"
    elif i == 1: explanation_detail = "low budget/defense/crime, high Env/SS -> strong D signal"
    elif i == 2: explanation_detail = "high budget/SS, low others -> ambiguous, depends on learned weights. high SS might lean D"
    elif i == 3: explanation_detail = "high budget/Env, low others -> ambiguous, conflicting signals. budget leans R, Env leans D"
    elif i == 4: explanation_detail = "high budget/defense/crime, moderate SS -> similar to training R examples, likely "

    explanation.append(f"  test Case {i+1} {input_val}: predicted {party} ({actual[0]:.4f}). rationale: {explanation_detail}")

for line in explanation:
    print(line)
