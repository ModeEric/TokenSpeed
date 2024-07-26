import time
from typing import List, Dict
from transformers import AutoTokenizer

class TokenizerEfficiencyTool:
    def __init__(self, tokenizer_names: List[str]):
        self.tokenizers = {name: AutoTokenizer.from_pretrained(name) for name in tokenizer_names}
        
    def measure_efficiency(self, text: str) -> Dict[str, Dict[str, float]]:
        results = {}
        for name, tokenizer in self.tokenizers.items():
            start_time = time.time()
            tokens = tokenizer.encode(text)
            end_time = time.time()
            
            results[name] = {
                "time": end_time - start_time,
                "num_tokens": len(tokens),
                "compression_ratio": len(text) / len(tokens)
            }
        
        return results

    def compare_tokenizers(self, text: str):
        results = self.measure_efficiency(text)
        for name, metrics in results.items():
            print(f"Tokenizer: {name}")
            print(f"  Time taken: {metrics['time']:.4f} seconds")
            print(f"  Number of tokens: {metrics['num_tokens']}")
            print(f"  Compression ratio: {metrics['compression_ratio']:.2f}")
            print()

# Example usage
if __name__ == "__main__":
    tokenizer_names = ["bert-base-uncased", "gpt2", "roberta-base"]
    tool = TokenizerEfficiencyTool(tokenizer_names)
    
    sample_text = "This is a sample text to test tokenizer efficiency."
    tool.compare_tokenizers(sample_text)