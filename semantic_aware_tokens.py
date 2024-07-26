import nltk
from nltk.corpus import brown
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import multiprocessing

nltk.download('brown', quiet=True)
nltk.download('punkt', quiet=True)

def default_dict_float():
    return defaultdict(float)

class MultiprocessingSemanticTokenizer:
    def __init__(self, vector_size=100, max_token_length=10):
        self.vector_size = vector_size
        self.max_token_length = max_token_length
        self.token_vectors = {}
        self.context_scores = defaultdict(default_dict_float)
        self.token_pairs = defaultdict(float)

    def train(self, corpus, num_iterations=5):
        # Initialize with single characters
        for sentence in corpus:
            for char in ''.join(sentence).lower():
                if char not in self.token_vectors:
                    self.token_vectors[char] = np.random.randn(self.vector_size) / np.sqrt(self.vector_size)

        # Training iterations
        for iteration in tqdm(range(num_iterations), desc="Training iterations"):
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = list(tqdm(pool.imap(self._update_token_vectors_parallel, corpus), 
                                    total=len(corpus), desc="Processing sentences"))
            
            # Combine results from parallel processing
            for context_scores, token_pairs in results:
                for pair, scores in context_scores.items():
                    self.context_scores[pair].update(scores)
                for pair, score in token_pairs.items():
                    self.token_pairs[pair] += score

            # After each iteration, merge top scoring token pairs
            self._merge_top_pairs()

    def _update_token_vectors_parallel(self, sentence):
        tokens = list(''.join(sentence).lower())
        local_context_scores = defaultdict(default_dict_float)
        local_token_pairs = defaultdict(float)

        for i, token in enumerate(tokens):
            if token not in self.token_vectors:
                continue  # Skip tokens not in our vocabulary
            context = self._get_context(tokens, i)
            for context_token in context:
                if context_token not in self.token_vectors:
                    continue  # Skip context tokens not in our vocabulary
                pair = tuple(sorted([token, context_token]))
                semantic_score = np.dot(self.token_vectors[token], self.token_vectors[context_token])
                semantic_score = np.clip(semantic_score, -5, 5)  # Prevent extreme values
                local_context_scores[pair][f"{i}_{tokens.index(context_token)}"] += semantic_score
                local_token_pairs[pair] += semantic_score

        return local_context_scores, local_token_pairs

    def _get_context(self, tokens, index, window_size=5):
        start = max(0, index - window_size)
        end = min(len(tokens), index + window_size + 1)
        return tokens[start:index] + tokens[index+1:end]

    def _merge_top_pairs(self, top_k=1000):
        sorted_pairs = sorted(self.token_pairs.items(), key=lambda x: x[1], reverse=True)
        for (token1, token2), score in tqdm(sorted_pairs[:top_k], desc="Merging top pairs"):
            new_token = token1 + token2
            if len(new_token) <= self.max_token_length and new_token not in self.token_vectors:
                # Create a new vector for the merged token
                self.token_vectors[new_token] = (self.token_vectors[token1] + self.token_vectors[token2]) / 2
                # Update context scores for the new token
                for context in set(self.context_scores[(token1, token2)].keys()) | set(self.context_scores[(token2, token1)].keys()):
                    new_pair = tuple(sorted([new_token, context]))
                    for context_id in self.context_scores[(token1, token2)]:
                        self.context_scores[new_pair][context_id] += (
                            self.context_scores[(token1, token2)][context_id] +
                            self.context_scores[(token2, token1)][context_id]
                        ) / 2
        # Reset token pairs for next iteration
        self.token_pairs = defaultdict(float)

    def tokenize(self, text):
        tokens = list(text.lower())
        result = []
        i = 0
        while i < len(tokens):
            best_token = tokens[i]
            best_score = 0
            for j in range(i + 1, min(i + self.max_token_length, len(tokens) + 1)):
                candidate = ''.join(tokens[i:j])
                if candidate in self.token_vectors:
                    context = self._get_context(tokens, i, window_size=5)
                    score = np.mean([np.dot(self.token_vectors[candidate], self.token_vectors[c]) 
                                     for c in context if c in self.token_vectors])
                    if score > best_score:
                        best_token = candidate
                        best_score = score
            result.append((best_token, best_score))
            i += len(best_token)
        return result

# Example usage
if __name__ == "__main__":
    tokenizer = MultiprocessingSemanticTokenizer()
    corpus = brown.sents()  # Increased subset of Brown corpus for training
    tokenizer.train(corpus, num_iterations=5)
    
    sample_text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer.tokenize(sample_text)
    
    print("Tokenization result:")
    for token, score in tokens:
        print(f"{token}: {score:.4f}")

    print("\nNumber of unique tokens:", len(tokenizer.token_vectors))