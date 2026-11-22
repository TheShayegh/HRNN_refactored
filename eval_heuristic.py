'''
This file's purpose is to evaluate a heuristic by calculating f1 score and accuracy by using a function named evaluate_heuristic 
'''
from library.utils import read_entries

if __name__ == "__main__":
    HEURISTIC_PRED_SENTENCES = 'data\\en_new_notation_pseudo_labels\\test.txt'
    GROUND_TRUTH_SENTENCES = 'data\\en_new_notation_ground_truths\\test.txt'
    ground_truth_entries = read_entries(GROUND_TRUTH_SENTENCES)
    heuristic_entries = read_entries(HEURISTIC_PRED_SENTENCES)
    fscore, acc = eval(ground_truth_entries, heuristic_entries)
    print( " __________________________________")
    print(f"| Test:")
    print(f"|     F1:       {fscore}")
    print(f"|     Accuracy: {acc}")
    print( "|__________________________________")