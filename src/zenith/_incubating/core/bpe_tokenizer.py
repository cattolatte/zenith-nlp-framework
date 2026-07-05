import re
from collections import defaultdict
import json

class BPETokenizer:
    def __init__(self):

        self.vocab = {}
        self.merges = {}
        self.token_to_id = {}
        self.id_to_token = {}

    def get_stats(self, vocab):
 
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        """
        Merges a pair of symbols in the vocabulary.

        Args:
            pair (tuple): The pair of symbols to merge.
            v_in (dict): The input vocabulary.

        Returns:
            dict: The vocabulary with the pair merged.
        """
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, corpus, vocab_size):
        """
        Trains the tokenizer on a given corpus.

        Args:
            corpus (list of str): A list of sentences to train on.
            vocab_size (int): The desired size of the vocabulary.
        """
        # 1. Pre-tokenization and initial vocabulary creation
        vocab = defaultdict(int)
        for text in corpus:
            words = text.strip().split()
            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1

        # 2. Build the base vocabulary of characters
        alphabet = set()
        for word in vocab:
            alphabet.update(word.split())
        
        # 3. Iteratively learn merge rules
        num_merges = vocab_size - len(alphabet)
        merges = {}
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best_pair, vocab)
            merges[best_pair] = i

        self.vocab = sorted(alphabet)
        self.merges = merges

        # Build final vocabulary and token mappings
        final_vocab = list(alphabet)
        for pair in merges:
            final_vocab.append("".join(pair))
        
        self.token_to_id = {token: i for i, token in enumerate(final_vocab)}
        self.id_to_token = {i: token for token, i in enumerate(final_vocab)}

    def tokenize(self, text):
        """
        Tokenizes a given text using the learned merge rules.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list of str: The list of subword tokens.
        """
        pre_tokenized_words = []
        words = text.strip().split()
        for word in words:
            pre_tokenized_words.append(' '.join(list(word)) + ' </w>')

        for pair, _ in sorted(self.merges.items(), key=lambda x: x[1]):
            new_words = []
            bigram = re.escape(' '.join(pair))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            for word in pre_tokenized_words:
                new_words.append(p.sub(''.join(pair), word))
            pre_tokenized_words = new_words

        tokens = ' '.join(pre_tokenized_words).split()
        return [self.token_to_id.get(token, -1) for token in tokens] # Return IDs, -1 for OOV
    
    def detokenize(self, token_ids):
        """
        Converts a list of token IDs back into a string.
        
        Args:
            token_ids (list of int): A list of token IDs.
            
        Returns:
            str: The reconstructed text.
        """
        tokens = [self.id_to_token.get(idx, "") for idx in token_ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()

    def save(self, filepath):
        """
        Saves the tokenizer's merges and vocabulary to a file.

        Args:
            filepath (str): The path to the file where the tokenizer will be saved.
        """
        model_data = {
            "merges": { ' '.join(k): v for k, v in self.merges.items()},
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

    def load(self, filepath):
        """
        Loads the tokenizer's merges and vocabulary from a file.

        Args:
            filepath (str): The path to the file from which to load the tokenizer.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)

        self.merges = {tuple(k.split(' ')): v for k, v in model_data["merges"].items()}
        self.token_to_id = model_data["token_to_id"]
        # JSON keys are strings, so convert id_to_token keys back to integers
        self.id_to_token = {int(k): v for k, v in model_data["id_to_token"].items()}
