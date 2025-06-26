import os
import yaml
import psutil # For get_available_cores and use_gpu (memory part)

# Define GB for clarity, useful in other modules too
GB = 1024 * 1024 * 1024

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_path} not found. Returning empty config.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration from {config_path}: {e}")
        return {}

def get_available_cores() -> int:
    """
    Returns the number of available logical CPU cores.
    Defaults to 1 if undetermined.
    """
    cores = os.cpu_count()
    if cores is None:
        try:
            # psutil.cpu_count(logical=True) might be more reliable on some systems
            cores = psutil.cpu_count(logical=True)
        except AttributeError: # If psutil is not installed or old version
             pass # Keep cores as None
    return cores if cores else 1


def use_gpu(memory_threshold_gb=6) -> bool:
    """
    Placeholder function to determine if GPU should be used.
    Checks an environment variable and optionally system memory as a proxy.
    """
    # Priority to environment variable
    env_gpu = os.getenv("SURVEYASSISTANT_USE_GPU", "").lower()
    if env_gpu == "1" or env_gpu == "true":
        print("Utils: GPU usage enabled via SURVEYASSISTANT_USE_GPU environment variable.")
        return True
    if env_gpu == "0" or env_gpu == "false":
        print("Utils: GPU usage disabled via SURVEYASSISTANT_USE_GPU environment variable.")
        return False

    # Fallback: Simple check for "enough" system RAM as a very rough proxy for a decent system
    # This is NOT a real GPU check. A real check involves libraries like torch.cuda.is_available()
    # or similar for other ML frameworks.
    try:
        available_ram_gb = psutil.virtual_memory().available / GB
        if available_ram_gb >= memory_threshold_gb:
            # This is a heuristic: systems with more RAM might also have a GPU.
            # print(f"Utils: Sufficient system RAM ({available_ram_gb:.1f}GB available) detected, tentatively enabling GPU usage if available by ML libs.")
            # The actual decision to use GPU layers in llama.cpp or torch is made by those libraries.
            # This function just signals a preference or checks a high-level config.
            # For now, let's assume this means "it's okay to try using a GPU if one is found by the ML lib"
            return True # Tentatively true, actual usage depends on ML lib detecting a GPU
    except Exception as e:
        # print(f"Utils: Could not check system memory for GPU heuristic: {e}")
        pass # Fallback to default (False) if psutil fails or not available

    # Default if no env var and RAM check is inconclusive or below threshold
    # print("Utils: Defaulting to no GPU usage (no explicit override and/or low memory).")
    return False


def ensure_dir_exists(directory_path: str):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Utils: Created directory: {directory_path}")
        except OSError as e:
            print(f"Utils: Error creating directory {directory_path}: {e}")
            raise # Re-raise the exception if directory creation fails critically

if __name__ == '__main__':
    print("Testing utility functions...")

    # Test config loading (requires a dummy config.yaml)
    print("\nTesting load_config:")
    # Create a dummy config for testing this script directly
    dummy_config_content = """
models:
  embedding:
    name: "bge-large-en-v1.5"
    path: "/models/bge-large"
  llm:
    name: "Qwen2.5-7B-Chat"
    path: "/models/qwen2_5-7b-chat-Q8_0.gguf"
"""
    with open("temp_config.yaml", "w") as f:
        f.write(dummy_config_content)

    config = load_config("temp_config.yaml")
    if config:
        print("Loaded config content:", config.get("models", {}).get("llm", {}))

    config_bad = load_config("non_existent_config.yaml") # Test non-existent
    assert config_bad == {}

    with open("temp_invalid_config.yaml", "w") as f: # Test invalid YAML
        f.write("models: \n  llm: path: this_is_bad_indentation")
    config_invalid = load_config("temp_invalid_config.yaml")
    assert config_invalid == {}

    # Clean up dummy files
    os.remove("temp_config.yaml")
    os.remove("temp_invalid_config.yaml")

    print("\nTesting get_available_cores:")
    cores = get_available_cores()
    print(f"Available CPU cores: {cores}")
    assert cores >= 1

    print("\nTesting use_gpu:")
    # Test without env var
    gpu_pref = use_gpu()
    print(f"GPU usage preference (default): {gpu_pref}")

    # Test with env var
    os.environ["SURVEYASSISTANT_USE_GPU"] = "1"
    gpu_pref_env_true = use_gpu()
    print(f"GPU usage preference (SURVEYASSISTANT_USE_GPU=1): {gpu_pref_env_true}")
    assert gpu_pref_env_true is True

    os.environ["SURVEYASSISTANT_USE_GPU"] = "false"
    gpu_pref_env_false = use_gpu()
    print(f"GPU usage preference (SURVEYASSISTANT_USE_GPU=false): {gpu_pref_env_false}")
    assert gpu_pref_env_false is False

    del os.environ["SURVEYASSISTANT_USE_GPU"] # Clean up

    print("\nTesting ensure_dir_exists:")
    test_dir = "temp_test_dir/subdir"
    ensure_dir_exists(test_dir)
    assert os.path.exists(test_dir)
    # Clean up test directory
    import shutil
    shutil.rmtree("temp_test_dir")
    print(f"Cleaned up {test_dir}")

    print("\nUtility functions test finished.")


# --- Adaptive Inference (Conceptual Implementation) ---
# Attempt to import heavy libraries, set flags
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE_UTIL = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE_UTIL = False

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE_UTIL = True
except ImportError:
    LLAMA_CPP_AVAILABLE_UTIL = False

# Assuming torch might be a dependency for a fact-checking model or other utilities
# try:
#     import torch
#     TORCH_AVAILABLE_UTIL = True
# except ImportError:
#     TORCH_AVAILABLE_UTIL = False


def adaptive_inference(text_input: str, model_type: str, prompt: str = None, model_name_override: str = None):
    """
    Conceptual function to perform adaptive inference based on model type and resources.
    This is a high-level simulation as actual model loading and execution is complex.

    Args:
        text_input (str): The primary text input for the model.
        model_type (str): Type of model, e.g., "embedding", "llm", "fact_check".
        prompt (str, optional): Specific prompt for LLM or fact-checking tasks.
        model_name_override (str, optional): Specific model name to use, overriding config default for this call.

    Returns:
        Varies by model_type:
        - "embedding": Simulated embedding vector (list of floats or numpy array).
        - "llm": Simulated text response from LLM (string).
        - "fact_check": Simulated fact-checking result (dict).
    """
    print(f"\nAdaptiveInference: Request for model_type='{model_type}', input_text='{text_input[:50]}...'")
    config_data = load_config() # Reload config, or assume it's passed/global
    if not config_data:
        print("AdaptiveInference: Error - Configuration not loaded.")
        return None

    # Resource check (simplified)
    gpu_preferred = use_gpu() # Checks env var or basic RAM heuristic
    available_cores = get_available_cores()

    print(f"AdaptiveInference: Resource check - GPU preferred: {gpu_preferred}, CPU cores: {available_cores}")

    if model_type == "embedding":
        model_config = config_data.get("models", {}).get("embedding", {})
        model_to_use = model_name_override if model_name_override else model_config.get("name", "bge-large-en-v1.5")
        model_path = model_config.get("path", model_to_use) # Path can be name for sentence-transformers
        dimension = 1024 if "large" in model_to_use.lower() else 768 # Simplified dimension logic

        print(f"AdaptiveInference (Embedding): Model: {model_to_use}, Path: {model_path}, Dim: {dimension}")

        if SENTENCE_TRANSFORMER_AVAILABLE_UTIL:
            print(f"AdaptiveInference (Embedding): Attempting to use SentenceTransformer for {model_to_use}.")
            # Actual model loading and encoding would happen here.
            # For simulation, we return a representative embedding.
            # Example:
            #   model = SentenceTransformer(model_path)
            #   embedding = model.encode([text_input])[0]
            #   return embedding
            print(f"AdaptiveInference (Embedding): SIMULATING embedding generation for '{text_input[:30]}...'.")
            import numpy as np
            sim_embedding = np.random.rand(dimension).astype('float32')
            return (sim_embedding / np.linalg.norm(sim_embedding)).tolist() # Normalized
        else:
            print(f"AdaptiveInference (Embedding): SentenceTransformer not available. SIMULATING embedding for '{model_to_use}'.")
            import numpy as np
            sim_embedding = np.random.rand(dimension).astype('float32')
            return (sim_embedding / np.linalg.norm(sim_embedding)).tolist()


    elif model_type == "llm":
        model_config = config_data.get("models", {}).get("llm", {})
        model_to_use = model_name_override if model_name_override else model_config.get("name", "Qwen2.5-7B-Chat")
        model_path = model_config.get("path", "models/qwen2_5-7b-chat-Q8_0.gguf")
        context_size = model_config.get("context_size", 4096)
        n_gpu_layers_config = model_config.get("n_gpu_layers", 0)

        # Adaptive n_gpu_layers (conceptual)
        effective_n_gpu_layers = n_gpu_layers_config if gpu_preferred else 0

        print(f"AdaptiveInference (LLM): Model: {model_to_use}, Path: {model_path}, Ctx: {context_size}, GPU Layers: {effective_n_gpu_layers} (Config: {n_gpu_layers_config}, GPU Preferred: {gpu_preferred})")

        if LLAMA_CPP_AVAILABLE_UTIL:
            print(f"AdaptiveInference (LLM): Attempting to use LlamaCPP for {model_to_use}.")
            # Actual LlamaCPP loading and inference would happen here.
            # Example:
            #   llm = Llama(model_path=model_path, n_ctx=context_size, n_gpu_layers=effective_n_gpu_layers, n_threads=available_cores)
            #   full_prompt = f"{prompt}\n{text_input}" # Construct prompt
            #   output = llm(full_prompt, max_tokens=150) # Example params
            #   return output['choices'][0]['text']
            print(f"AdaptiveInference (LLM): SIMULATING LLM response for prompt='{prompt}', text='{text_input[:30]}...'.")
            return f"[Simulated LLM Response for '{prompt}']: Based on '{text_input[:30]}...', the key aspects are A, B, and C."
        else:
            print(f"AdaptiveInference (LLM): LlamaCPP not available. SIMULATING LLM response for '{model_to_use}'.")
            return f"[Simulated LLM Response (LlamaCPP N/A) for '{prompt}']: Input text started with '{text_input[:30]}...'."


    elif model_type == "fact_check":
        model_config = config_data.get("models", {}).get("fact_check", {})
        model_to_use = model_name_override if model_name_override else model_config.get("name", "FactLLM")
        model_path = model_config.get("path", "models/factllm-3b")

        # Conceptual: ONNX runtime for CPU as per README example if GPU not used.
        inference_mode = "GPU (simulated)" if gpu_preferred else "ONNX CPU (simulated)"
        print(f"AdaptiveInference (FactCheck): Model: {model_to_use}, Path: {model_path}, Mode: {inference_mode}")

        # Actual model loading (e.g. HuggingFace Transformers, ONNX runtime) would happen here.
        # For simulation:
        print(f"AdaptiveInference (FactCheck): SIMULATING fact-check for text='{text_input[:30]}...'.")
        import random
        consistency_score = round(random.uniform(0.6, 0.95), 2)
        return {
            "text_checked": text_input[:100] + "...",
            "consistency_score": consistency_score,
            "issues": [] if consistency_score > 0.75 else [{"issue": "Simulated low confidence area found."}],
            "model_used_sim": model_to_use,
            "inference_mode_sim": inference_mode
        }

    else:
        print(f"AdaptiveInference: Error - Unknown model_type '{model_type}'.")
        return None

# Example usage for adaptive_inference (can be run if this file is executed)
if __name__ == '__main__':
    print("\n--- Testing Adaptive Inference ---")

    # Ensure a dummy config.yaml exists for this test if not present globally
    if not os.path.exists("config.yaml"):
        dummy_cfg_content = """
models:
  embedding:
    name: "bge-large-en-v1.5"
    path: "models/bge-large-en-v1.5" # or just the name for sentence-transformers
  llm:
    name: "Qwen2.5-7B-Chat"
    path: "models/qwen2_5-7b-chat-Q8_0.gguf"
    context_size: 2048
  fact_check:
    name: "FactLLM"
    path: "models/factllm-3b"
"""
        with open("config.yaml", "w") as f:
            f.write(dummy_cfg_content)
        print("Created dummy config.yaml for adaptive_inference test.")

    sample_text = "This is a sample text for model processing about artificial intelligence."

    print("\nTesting Embedding Model via Adaptive Inference:")
    embedding_result = adaptive_inference(sample_text, "embedding")
    if embedding_result:
        print(f"Embedding Result (first 5 dims): {embedding_result[:5]}..., Length: {len(embedding_result)}")

    print("\nTesting LLM via Adaptive Inference:")
    llm_result = adaptive_inference(sample_text, "llm", prompt="Summarize the following text:")
    if llm_result:
        print(f"LLM Result: {llm_result}")

    print("\nTesting Fact Check Model via Adaptive Inference:")
    fact_check_result = adaptive_inference(sample_text, "fact_check")
    if fact_check_result:
        print(f"Fact Check Result: {fact_check_result}")

    print("\n--- Adaptive Inference Test Finished ---")
    # Clean up dummy config if it was created by this test block
    # if os.path.exists("config.yaml") and "dummy_cfg_content" in locals():
    #     os.remove("config.yaml")
    #     print("Removed dummy config.yaml")
