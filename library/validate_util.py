

import csv

def validate(model, data, entries, name,validate_csv_file,device):
    loss, validation_entries = model.predict(
        data,
        [entry.get_words() for entry in entries],
        device=device,
    )

    fscore, acc = eval(entries,validation_entries)
    with open(validate_csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name,loss,fscore,acc])

    return loss, fscore, acc

def evaluate_f1_score(ground_truth_spans, predicted_spans):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for true_spans, pred_spans in zip(ground_truth_spans, predicted_spans):
        true_set = set(true_spans)
        pred_set = set(pred_spans)
        true_positive += len(true_set.intersection(pred_set))
        false_positive += len(pred_set - true_set)
        false_negative += len(true_set - pred_set)
     
    precision =  100 * true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = 100 * true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return f1

def evaluate_accuracy_score(ground_truth_spans, predicted_spans):
    # Calculate the number of correct tags and total tags
    correct_tags = 0
    total_tags = 0
    for i in range(len(ground_truth_spans)):
        gt_span = ground_truth_spans[i]
        pred_span = predicted_spans[i]
        b_gt = set([span[0] for span in gt_span])
        b_pred = set([span[0] for span in pred_span])
        total_tags += ground_truth_spans[i][-1][1]
        # Total words - missed Bs in both 
        correct_tags += len(b_gt ^ b_pred) 
    # Calculate accuracy
    if total_tags > 0:
        accuracy = ( (total_tags - correct_tags) / total_tags) * 100
    else:
        accuracy = 0
    return accuracy 

def eval(gt_entries, pred_entries):
    ground_truth_spans = [sentence.get_chunk_spans() for sentence in gt_entries]
    predicted_spans = [sentence.get_chunk_spans() for sentence in pred_entries]
    # We use the spans to calculate the f1 score and accuracy
    f1  = evaluate_f1_score(ground_truth_spans, predicted_spans)
    accuracy = evaluate_accuracy_score(ground_truth_spans, predicted_spans)

    return f1, accuracy
            
