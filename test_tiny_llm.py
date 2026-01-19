import numpy as np
import time

def simulate_llm_inference():
    print("--- KnowledgeOS Small LLM Test (TinyLlama-1.1B Emulation) ---")
    
    # TinyLlama parameters
    hidden_size = 2048
    intermediate_size = 5632
    
    try:
        # 1. Simulate Weight Loading (Memory Mapping)
        print(f"Allocating Virtual Memory for Layer Weights...")
        weights = np.random.randn(hidden_size, intermediate_size).astype(np.float16)
        inputs = np.random.randn(1, hidden_size).astype(np.float16)
        
        # 2. Simulate Forward Pass (Matrix Multiply)
        print("Computing Transformer Layer...")
        start_time = time.time()
        output = np.matmul(inputs, weights)
        end_time = time.time()
        
        # 3. Calculate Performance
        print(f"Inference Time: {end_time - start_time:.4f} seconds")
        print(f"Memory Usage: {weights.nbytes / 1024 / 1024:.2f} MB for one layer weights")
        print("Status: SUCCESS - OS handles Float16 and Large MatMul")
        
    except MemoryError:
        print("Status: FAILED - Out of Memory. Increase SWAP or RAM.")
    except Exception as e:
        print(f"Status: ERROR - {e}")

if __name__ == "__main__":
    simulate_llm_inference()
