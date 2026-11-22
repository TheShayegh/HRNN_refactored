from library.labelled_entry import LabelledEntry
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures

def ensemble_by_tau(entries, output_tau, weights=None, beta=1):
    if weights is None:
        weights = [1]*len(entries)
    words = entries[0].get_words()

    hitcount = defaultdict(lambda: 0)
    for entry, weight in zip(entries, weights):
        for chunk in entry.get_chunk_spans():
            hitcount[chunk] += weight / (output_tau + (beta**2)*len(entry.chunks))

    DPtable = [[[(-float('inf'), None) for tau in range(l+1)] for b in range(len(words)-l+1)] for l in range(len(words)+1)]
    for l in range(1, len(words)+1):
        for b in range(len(words)-l+1):
            chunk = (b, b+l)
            DPtable[l][b][1] = (hitcount[chunk], [chunk])
            for tau in range(2, l+1):
                for j in range(1, l):
                    for left_tau in range(1, j+1):
                        right_tau = tau-left_tau
                        if right_tau<=0 or right_tau>l-j:
                            continue
                        left = DPtable[j][b][left_tau]
                        right = DPtable[l-j][b+j][right_tau]
                        if left[0]+right[0] > DPtable[l][b][tau][0]:
                            DPtable[l][b][tau] = (left[0]+right[0], left[1]+right[1])

    return LabelledEntry.load_from_chunk_spans(words, DPtable[-1][0][output_tau][1])


def fb(entry1, entry2, beta=1):
    intersection = set(entry1.get_chunk_spans()).intersection(set(entry2.get_chunk_spans()))
    return (1+beta**2)*len(intersection) / (len(entry1.chunks) + (beta**2)*len(entry2.chunks))


def ensemble(entries, weights=None, beta=1):
    words = entries[0].get_words()
    ensemble_by_taus = [ensemble_by_tau(entries, tau, weights=weights, beta=beta) for tau in range(1, len(words)+1)]
    fb_scores = [sum([fb(e, entry, beta) for entry in entries]) for e in ensemble_by_taus]
    best = max(zip(fb_scores, ensemble_by_taus), key=lambda x: x[0])[1]
    return best


def worker(i, individual_entries, weights, beta):
    output = ensemble(individual_entries, weights=weights, beta=beta)
    return i, output

def parallel_ensemble(individuals, weights=None, beta=1, progress_bar=True):
    bar = tqdm if progress_bar else lambda x, **args: x
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(worker, i, [individual[i] for individual in individuals], weights=weights, beta=beta) for i in range(len(individuals[0]))]
        outputs = [f.result() for f in bar(concurrent.futures.as_completed(futures), total=len(futures))]
        outputs = sorted(outputs, key=lambda x: x[0])
        outputs = list(map(lambda x: x[1], outputs))
    return outputs