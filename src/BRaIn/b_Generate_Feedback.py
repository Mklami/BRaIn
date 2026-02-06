import json
import sys
import hashlib
from pathlib import Path

# Add src directory to Python path
script_dir = Path(__file__).parent.parent.parent.absolute()
src_dir = script_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from tqdm import tqdm
from Utils import JSON_File_IO
from transformers import AutoTokenizer


def load_dataframe(file_path):
    return JSON_File_IO.load_JSON_to_Dataframe(file_path)


def load_json_to_dict(file_path):
    return JSON_File_IO.load_JSON_to_Dict(file_path)


def get_bug_id(bug):
    """
    Generate a unique identifier for a bug based on its key fields.
    This allows us to track which bugs have been processed.
    """
    # Use project, sub_project, version, and bug_title to create unique ID
    key_fields = f"{bug.get('project', '')}_{bug.get('sub_project', '')}_{bug.get('version', '')}_{bug.get('bug_title', '')}"
    return hashlib.md5(key_fields.encode()).hexdigest()


def is_bug_processed(bug, processed_bugs):
    """
    Check if a bug has already been processed by checking if es_results have LLM scores.
    A bug is considered processed if any method in es_results has a relevance score.
    """
    es_results = bug.get('es_results', [])
    for result in es_results:
        methods = result.get('methods', {})
        # Check if any method has a string value (yes/no/possible) instead of just method body
        for method_name, method_value in methods.items():
            if isinstance(method_value, str) and method_value in ['yes', 'no', 'possible']:
                return True
    return False


def truncate_prompt(tokenizer, prompt, max_tokens=7500):
    """
    Truncate a prompt to fit within max_tokens.
    If the prompt is too long, it will be truncated from the end (code segment).
    """
    # Tokenize to check length
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return prompt
    
    # If too long, truncate from the end
    # Keep the beginning (instructions, bug report) and truncate the code segment
    truncated_tokens = tokens[:max_tokens]
    truncated_prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    # Add a note that truncation occurred
    if "[Code truncated]" not in truncated_prompt:
        # Try to add truncation marker before the last part
        truncated_prompt += "\n[Code truncated due to length limit]"
    
    return truncated_prompt


def llm_scoring(es_results, bug_title, bug_description, llm, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    user_role = {"role": "user", "content": ""}
    assistant_role = {"role": "assistant", "content": ""}

    user_text = """You are a helpful AI software engineer specializing in identifying buggy code segments given a bug report. Analyze the provided bug report and the JAVA code segment to determine if the code segment is responsible for causing the bug described in the bug report. You need to understand the functionality of the code segment and the details of the bug report to determine the relevance of the code segment to the bug report.

There are three possible outputs: 'yes', 'no', 'possible'.
-'yes': The code is responsible for the bug described in the bug report.
-'no': The code is NOT responsible for the bug described in the bug report.
-'possible': The code can be partially responsible for the bug described in the bug report.

Provide your output in JSON format like this sample: {"relevance": "yes"}.

Act like a rational software engineer and provide output. Avoid emotion and extra text other than JSON.

### 
Analyze the following bug report and code segment for relevance:"""

    instruction = '''Please determine if the code segment is responsible for the bug described in the bug report.'''

    bug_report = f'''Bug Report: \n- {bug_title} \n- {bug_description}'''

    for result in es_results:
        # file_url = result['file_url']
        # bm25_score = result['bm25_score']

        methods = result['methods']

        prompts = []

        # now, iterate over the key/value pairs of the methods in dictionary
        for method_name, method_body in methods.items():
            # add the method_body to the context

            code_context = f'''Code Segment: \n {method_body} '''

            # now, create the prompt
            prompt = user_text + '\n\n' + bug_report + '\n\n' + code_context + '\n\n' + instruction + '\n###'
            
            # Truncate prompt if it's too long (leave buffer for model output)
            # Use 7500 tokens max to leave room for response
            prompt = truncate_prompt(tokenizer, prompt, max_tokens=7500)

            chat = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""},
            ]

            template = tokenizer.apply_chat_template(chat, tokenize=False)
            prompts.append(template)

        outputs = llm.generate(prompts)

        # zip through outputs and methods
        for output, method_name in zip(outputs, methods.keys()):
            # prompt = output.prompt
            response = output.outputs[0].text

            is_relevant = 'no'

            # check if the response contains 'yes', 'no'
            if 'yes' in response:
                is_relevant = 'yes'
            elif 'no' in response:
                is_relevant = 'no'

            # is_relevant = json.loads(response)['relevance']

            # add the score to the es_results
            result['methods'][method_name] = is_relevant

    return es_results


import html
import os
import json

# ============================================================================
# CONFIGURATION: Update these paths before running
# ============================================================================
# Model path: Use a GPTQ quantized model from HuggingFace
# RECOMMENDED: Use HuggingFace model ID directly (more reliable with vLLM)
MODEL_PATH = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
# Alternative: Use local model (may have compatibility issues with some vLLM versions)
# MODEL_PATH = str(Path(__file__).parent / "Mistral-7B-Instruct-v0.2-GPTQ")

if __name__ == '__main__':
    # Validate model path exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")
    
    config_path = os.path.join(MODEL_PATH, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in model directory: {config_path}")
    
    # Validate config.json has required fields and inspect it
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Config file found. Model type: {config.get('model_type', 'UNKNOWN')}")
        
        required_fields = ['hidden_size', 'num_attention_heads', 'model_type']
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            print(f"WARNING: config.json is missing fields: {missing_fields}")
            print(f"Available fields in config: {list(config.keys())[:10]}...")  # Show first 10 fields
        else:
            # Calculate head_dim to verify it can be computed
            hidden_size = config.get('hidden_size')
            num_heads = config.get('num_attention_heads')
            if hidden_size and num_heads:
                head_dim = hidden_size // num_heads
                print(f"Model config validated: hidden_size={hidden_size}, "
                      f"num_attention_heads={num_heads}, "
                      f"head_dim={head_dim}")
            else:
                print(f"WARNING: Cannot calculate head_dim - hidden_size={hidden_size}, num_attention_heads={num_heads}")
        
        # Check for GPTQ-specific fields
        if 'quantization_config' in config:
            print(f"Quantization config found: {config['quantization_config']}")
    except json.JSONDecodeError as e:
        raise ValueError(f"config.json is not valid JSON: {e}")
    except Exception as e:
        print(f"Warning: Could not validate config.json: {e}")
    
    print(f"Loading model from: {MODEL_PATH}")
    
    # Check for required model files
    required_files = ['config.json', 'tokenizer_config.json']
    gptq_files = [f for f in os.listdir(MODEL_PATH) if 'gptq' in f.lower() or f.endswith('.safetensors') or f.endswith('.bin')]
    print(f"Found {len(gptq_files)} potential model weight files in directory")
    if len(gptq_files) == 0:
        print("WARNING: No model weight files found! Model may be incomplete.")
    else:
        print(f"Model weight files: {gptq_files[:5]}...")  # Show first 5
    
    # GPTQ models require dtype="half" (float16), not bfloat16
    # Try different loading strategies for GPTQ models
    # Note: max_model_len is set to 8192 (Mistral-7B's context length)
    # Prompts are truncated to 7500 tokens to leave room for responses
    MAX_MODEL_LEN = 8192
    
    # Check if MODEL_PATH is a local path that might have compatibility issues
    is_local_path = os.path.exists(MODEL_PATH) and os.path.isdir(MODEL_PATH)
    
    # Strategy 1: Try HuggingFace first if local model has known issues
    # (HuggingFace models are more likely to work due to better compatibility)
    if is_local_path:
        print("Local model path detected. Will try HuggingFace as fallback if local loading fails.")
    
    # Strategy 1: Let vLLM auto-detect GPTQ (recommended for local models)
    llm = None
    errors = []
    try:
        print("Attempting to load with auto-detection (dtype=half)...")
        llm = LLM(model=MODEL_PATH, dtype="half",
                  max_model_len=MAX_MODEL_LEN, trust_remote_code=True)
        print("Model loaded successfully with auto-detection!")
    except Exception as e1:
        errors.append(("auto-detect", e1))
        print(f"Auto-detection failed: {type(e1).__name__}: {str(e1)[:150]}")
        
        # If head_dim error, try gptq_marlin (faster, more compatible) or HuggingFace
        if 'head_dim' in str(e1).lower() or 'nonetype' in str(e1).lower():
            print("\nDetected head_dim error - trying gptq_marlin (more compatible)...")
            try:
                print("Trying with gptq_marlin quantization (recommended for GPTQ models)...")
                llm = LLM(model=MODEL_PATH, quantization="gptq_marlin", dtype="half",
                          max_model_len=MAX_MODEL_LEN, trust_remote_code=True)
                print("Model loaded successfully with gptq_marlin!")
            except Exception as marlin_error:
                print(f"gptq_marlin failed: {marlin_error}")
                errors.append(("gptq_marlin", marlin_error))
                
                # Try HuggingFace as fallback
                if MODEL_PATH.startswith('/') or not MODEL_PATH.startswith('TheBloke/'):
                    print("\nTrying HuggingFace model as alternative...")
                    hf_model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
                    try:
                        print(f"Loading from HuggingFace: {hf_model_id}")
                        llm = LLM(model=hf_model_id, quantization="gptq_marlin", dtype="half",
                                  max_model_len=MAX_MODEL_LEN, trust_remote_code=True)
                        print(f"Successfully loaded from HuggingFace: {hf_model_id}!")
                        # Update MODEL_PATH for tokenizer loading
                        MODEL_PATH = hf_model_id
                    except Exception as hf_error:
                        print(f"HuggingFace loading failed: {hf_error}")
                        errors.append(("huggingface", hf_error))
        
        # If HuggingFace didn't work or wasn't tried, continue with other strategies
        if llm is None:
            # Strategy 2: Explicit GPTQ quantization (lowercase)
            try:
                print("Trying with explicit 'gptq' quantization (dtype=half)...")
                llm = LLM(model=MODEL_PATH, quantization="gptq", dtype="half",
                          max_model_len=MAX_MODEL_LEN, trust_remote_code=True)
                print("Model loaded successfully with explicit GPTQ!")
            except Exception as e2:
                errors.append(("explicit_gptq", e2))
                print(f"Explicit GPTQ failed: {type(e2).__name__}: {str(e2)[:150]}")
                
                # Strategy 3: Try gptq_marlin (more compatible)
                if 'head_dim' in str(e2).lower():
                    print("\nTrying gptq_marlin quantization (more compatible with vLLM)...")
                    try:
                        llm = LLM(model=MODEL_PATH, quantization="gptq_marlin", dtype="half",
                                  max_model_len=MAX_MODEL_LEN, trust_remote_code=True)
                        print("Model loaded successfully with gptq_marlin!")
                    except Exception as marlin_error2:
                        errors.append(("gptq_marlin", marlin_error2))
                        print(f"gptq_marlin failed: {marlin_error2}")
                        
                        # Try HuggingFace if not already tried
                        if (MODEL_PATH.startswith('/') or not MODEL_PATH.startswith('TheBloke/')) and is_local_path:
                            print("\nTrying HuggingFace model as alternative...")
                            hf_model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
                            try:
                                print(f"Loading from HuggingFace: {hf_model_id}")
                                llm = LLM(model=hf_model_id, quantization="gptq_marlin", dtype="half",
                                          max_model_len=MAX_MODEL_LEN, trust_remote_code=True)
                                print(f"Successfully loaded from HuggingFace: {hf_model_id}!")
                                MODEL_PATH = hf_model_id
                            except Exception as hf_error2:
                                errors.append(("huggingface", hf_error2))
                                print(f"HuggingFace loading failed: {hf_error2}")
                
                # Strategy 4: Try with float16 explicitly
                if llm is None:
                    try:
                        print("Trying with explicit float16 dtype...")
                        import torch
                        llm = LLM(model=MODEL_PATH, quantization="gptq", dtype=torch.float16,
                                  max_model_len=MAX_MODEL_LEN, trust_remote_code=True)
                        print("Model loaded successfully with torch.float16!")
                    except Exception as e3:
                        errors.append(("torch_float16", e3))
                        print(f"Explicit float16 failed: {type(e3).__name__}: {str(e3)[:150]}")
    
    # If all strategies failed, raise error with summary
    if llm is None:
        print(f"\n{'='*80}")
        print("All loading strategies failed. Error summary:")
        print(f"{'='*80}")
        for strategy, error in errors:
            print(f"{strategy}: {type(error).__name__}: {str(error)[:200]}")
        
        raise RuntimeError(
            "Failed to load model with all strategies.\n\n"
            "The 'head_dim is None' error suggests vLLM cannot parse the local model config,\n"
            "even though config.json appears valid. This is a known vLLM/GPTQ compatibility issue.\n\n"
            "SOLUTIONS:\n"
            "1. Use HuggingFace model directly (recommended):\n"
            "   Change MODEL_PATH to: 'TheBloke/Mistral-7B-Instruct-v0.2-GPTQ'\n"
            "2. Re-download the model from HuggingFace to ensure all files are complete\n"
            "3. Try a different vLLM version: pip install vllm==0.6.3.post1\n"
            "4. Consider using AWQ quantization instead of GPTQ\n"
            "5. Use the base (non-quantized) model if quantization is not critical"
        )

    # Read from the cached output from a_Cache_initial_search_files.py
    # This should point to the output file(s) from the caching step
    sample_path = str(script_dir / "Output" / "Cache" / "Chunked_50" / "Cache_Res50_C1.json")

    # Output file path
    json_save_path = str(script_dir / "Output" / "Intelligent_Feedback")
    output_file = Path(json_save_path) / "Mistral_ZERO.json"
    
    # Load existing output if it exists (for resume functionality)
    processed_bugs = {}
    if output_file.exists():
        print(f"Found existing output file: {output_file}")
        print("Loading existing results to resume from checkpoint...")
        try:
            existing_bugs = load_json_to_dict(str(output_file))
            # Create a mapping of bug IDs to their processed state
            for bug in existing_bugs:
                bug_id = get_bug_id(bug)
                if is_bug_processed(bug, {}):
                    processed_bugs[bug_id] = bug
            print(f"Found {len(processed_bugs)} already processed bugs. Will skip these and continue from where it stopped.")
        except Exception as e:
            print(f"Warning: Could not load existing output file: {e}")
            print("Starting fresh...")
            processed_bugs = {}
    else:
        print("No existing output file found. Starting fresh...")

    # load the json to dictionary
    json_bugs = load_json_to_dict(sample_path)
    
    # Track progress
    total_bugs = len(json_bugs)
    processed_count = 0
    skipped_count = 0
    new_count = 0

    # iterate over the json array
    for idx, bug in enumerate(tqdm(json_bugs, desc="Processing JSON Bugs")):
        bug_id = get_bug_id(bug)
        
        # Check if this bug was already processed
        if bug_id in processed_bugs:
            # Update the bug with existing processed data
            json_bugs[idx] = processed_bugs[bug_id]
            skipped_count += 1
            continue
        
        # Check if bug is already processed (by checking es_results)
        if is_bug_processed(bug, {}):
            processed_bugs[bug_id] = bug
            skipped_count += 1
            continue
        
        # Process this bug
        try:
            bug_title = html.unescape(bug['bug_title'])
            bug_description = html.unescape(bug['bug_description'])
            project = bug['project']
            sub_project = bug['sub_project']
            version = bug['version']
            es_results = bug['es_results']

            score_llm_results = llm_scoring(es_results, bug_title, bug_description, llm=llm, model_path=MODEL_PATH)

            bug['es_results'] = score_llm_results
            processed_bugs[bug_id] = bug
            new_count += 1
            
            # Save progress after each bug (checkpoint)
            # This allows resuming if the script stops
            JSON_File_IO.save_Dict_to_JSON(json_bugs, json_save_path, "Mistral_ZERO.json")
            
        except Exception as e:
            print(f"\nError processing bug {idx+1}/{total_bugs} (project={bug.get('project', 'unknown')}): {e}")
            print("Saving progress so far...")
            # Save what we have so far
            JSON_File_IO.save_Dict_to_JSON(json_bugs, json_save_path, "Mistral_ZERO.json")
            # Re-raise to stop processing (or continue if you want to skip errors)
            raise

    # Final save
    print(f"\nProcessing complete!")
    print(f"Total bugs: {total_bugs}")
    print(f"Newly processed: {new_count}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"Processed in this run: {processed_count}")
    JSON_File_IO.save_Dict_to_JSON(json_bugs, json_save_path, "Mistral_ZERO.json")
    print(f"Results saved to: {output_file}")