import os
import re
import unicodedata

def clean_sentence(sentence):
    # Remove trailing spaces
    sentence = sentence.strip()
    # Normalize the sentence to NFKD form and remove unrecognizable characters
    sentence = ''.join(
        c for c in unicodedata.normalize('NFKD', sentence) 
        if not unicodedata.combining(c) and c.isprintable()
    )
    return sentence

def split_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    
    # Clean sentences
    sentences = [clean_sentence(sentence) for sentence in sentences]

    # Calculate the split index
    split_index = int(0.8 * len(sentences))

    # Split into two lists
    sentences_80 = sentences[:split_index]
    sentences_20 = sentences[split_index:]

    # Write the 80% sentences to one file
    with open('data/bhojpuri_sentences/train.txt', 'w', encoding='utf-8') as f:
        for sentence in sentences_80:
            f.write(sentence + '\n')

    # Write the 20% sentences to another file
    with open('data/bhojpuri_sentences/val.txt', 'w', encoding='utf-8') as f:
        for sentence in sentences_20:
            f.write(sentence + '\n')

# Example usage
input_file = 'data/bhojpuri-all.txt' # Change this to your input file path
split_file(input_file)
