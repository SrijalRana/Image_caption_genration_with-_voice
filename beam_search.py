
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def beam_search(model, tokenizer, photo, max_length, beam_width=5):
    start = [tokenizer.word_index['startseq']]
    sequences = [[start, 0.0]]

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            padded = pad_sequences([seq], maxlen=max_length, padding='post')
            preds = model.predict([photo, padded], verbose=0)[0]

            top_k = np.argsort(preds)[-beam_width:]

            for word in top_k:
                candidate = seq + [word]
                candidate_score = score - np.log(preds[word] + 1e-10)
                all_candidates.append([candidate, candidate_score])

        ordered = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered[:beam_width]

    best_seq = sequences[0][0]

    caption = []
    for idx in best_seq:
        word = tokenizer.index_word.get(idx)
        if word is None:
            continue
        if word == 'endseq':
            break
        caption.append(word)

    return ' '.join(caption[1:])
