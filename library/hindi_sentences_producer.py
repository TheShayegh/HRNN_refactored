from datasets import load_dataset
hiner = load_dataset('cfilt/HiNER-original')

text = '\n'.join([' '.join(sentence) for sentence in hiner['train']['tokens']])
output = "data/hindi_sentences/train.txt"
open(output, 'w').write(text)

text = '\n'.join([' '.join(sentence) for sentence in hiner['validation']['tokens']])
output = "data/hindi_sentences/val.txt"
open(output, 'w').write(text)
