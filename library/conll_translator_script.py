import sys

def extract_word():
    # Define a translation table to remove invisible characters like \xad
    invisible_chars = '\xad'
    translation_table = str.maketrans('', '', invisible_chars)
    
    file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    with open(file_path, 'r') as f:
        lines = f.readlines()
        sentences = []
        sentence = ''
        for line in lines:
            if line == '\n':
                sentences.append(sentence.strip())
                sentence = ''
                continue

            word = line.split('\t')[1].strip()
            # Remove invisible characters from the word
            word = word.translate(translation_table)
            if word in '.!?':
                sentence = sentence + word
            else:
                sentence = sentence + ' ' + word
        # Append the last sentence
        if sentence:
            sentences.append(sentence.strip())
    
    with open(output_file_path, 'w') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

extract_word()
