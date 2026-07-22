"""
onyx baseline test script

this script tests the baseline autoregressive generation loop,
measuring key performance metrics for the onyx inference engine.
"""

import sys

sys.path.insert(0, ".")

from onyx.engine import OnyxEngine, get_device_info
from onyx.evaluator import Evaluator


def main():
    print("onyx baseline generation test")
    
    device_info = get_device_info()
    print(f"device: {device_info['device']}")
    print(f"mlx available: {device_info['mlx_available']}")
    
    print("loading model...")
    print(f"model: mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    
    engine = OnyxEngine()
    load_time = engine.load_time
    print(f"model loaded in {load_time:.2f} seconds")
    
    prompt = "Explain the concept of recursion in one sentence."
    
    print(f"prompt: \"{prompt}\"")
    print("generating response...")
    
    response, metrics = engine.generate(
        prompt=prompt,
        max_tokens=128,
        temperature=0.0,
    )
    
    print(f"response: {response}")
    
    Evaluator.print_metrics("Baseline", metrics)
    print(f"time to load: {load_time:.2f} s")
    
    print("testing streaming generation...")
    print("streaming: ", end="", flush=True)
    
    stream_tokens = []
    stream_metrics = None
    for token_text, final_metrics in engine.stream_generate(
        prompt="What is 2 + 2?",
        max_tokens=32,
        temperature=0.0,
    ):
        if final_metrics is not None:
            stream_metrics = final_metrics
        else:
            print(token_text, end="", flush=True)
            stream_tokens.append(token_text)
    
    if stream_metrics:
        print(f"\nstreaming ttft: {stream_metrics['ttft'] * 1000:.1f} ms")
        print(f"streaming speed: {stream_metrics['tokens_per_second']:.1f} tokens/s")
    
    print("cache abstraction verification:")
    print(f"cache manager: {type(engine.cache_manager).__name__}")
    print(f"cache type: {type(engine.cache_manager.caches[0]).__name__}")
    print(f"number of layers: {engine.cache_manager.num_layers}")
    print(f"cache size (tokens): {engine.cache_manager.total_size()}")
    
    Evaluator.log_baseline(metrics, prompt, response, load_time)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
