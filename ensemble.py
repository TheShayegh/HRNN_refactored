from library.utils import read_entries
from library.ensemble import parallel_ensemble
from library.validate_util import eval


if __name__ == '__main__':
    gold = 'data/ger_ground_truths/test.txt'
    paths = [
        'experiments_outputs/2024-05-06_16:26:01.093882/output/german_test_predictions.txt',
        # 'experiments_outputs/2024-06-03_17:11:48.949143/output/german_test_predictions.txt',
        'experiments_outputs/2024-05-23_23:00:27.853236/output/german_test_predictions.txt',
        # 'experiments_outputs/2024-06-03_17:49:43.823706/output/ger_test_predictions.txt',
        'experiments_outputs/2024-06-07_19:45:29.415228/output/ger_test_predictions.txt',
    ]
    weights = None
    individuals = [read_entries(p) for p in paths]

    outputs = parallel_ensemble(individuals, weights, beta=100)
    f1, acc = eval(read_entries(gold), outputs)
    print("F1 Score: ", f1, "AND", "Accuracy =", acc)