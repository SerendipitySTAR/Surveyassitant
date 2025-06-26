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
