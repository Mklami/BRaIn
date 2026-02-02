import json
import sys
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

# ============================================================================
# CONFIGURATION: Update these paths before running
# ============================================================================
# Model path: Use a GPTQ quantized model from HuggingFace
# Example: "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
MODEL_PATH = "/home/m.lami/BRaIn/src/BRaIn/Mistral-7B-Instruct-v0.2-GPTQ"

if __name__ == '__main__':
    # Validate model path exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")
    
    config_path = os.path.join(MODEL_PATH, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in model directory: {config_path}")
    
    print(f"Loading model from: {MODEL_PATH}")
    
    # Try different loading strategies for GPTQ models
    # Strategy 1: Let vLLM auto-detect GPTQ (recommended)
    try:
        print("Attempting to load with auto-detection...")
        llm = LLM(model=MODEL_PATH, dtype="half",
                  max_model_len=8192, trust_remote_code=True)
        print("Model loaded successfully with auto-detection!")
    except Exception as e1:
        print(f"Auto-detection failed: {e1}")
        # Strategy 2: Explicit GPTQ quantization (lowercase)
        try:
            print("Trying with explicit 'gptq' quantization...")
            llm = LLM(model=MODEL_PATH, quantization="gptq", dtype="half",
                      max_model_len=8192, trust_remote_code=True)
            print("Model loaded successfully with explicit GPTQ!")
        except Exception as e2:
            print(f"Explicit GPTQ failed: {e2}")
            # Strategy 3: Try without dtype specification
            try:
                print("Trying without dtype specification...")
                llm = LLM(model=MODEL_PATH, quantization="gptq",
                          max_model_len=8192, trust_remote_code=True)
                print("Model loaded successfully without dtype!")
            except Exception as e3:
                print(f"All loading strategies failed. Last error: {e3}")
                raise RuntimeError("Failed to load model with all strategies. "
                                 "Please check:\n"
                                 "1. Model directory contains config.json\n"
                                 "2. Model files are complete and not corrupted\n"
                                 "3. vLLM version supports this GPTQ format\n"
                                 "4. Model architecture is compatible with vLLM") from e3

    # Read from the cached output from a_Cache_initial_search_files.py
    # This should point to the output file(s) from the caching step
    sample_path = str(script_dir / "Output" / "Cache" / "Chunked_50" / "Cache_Res50_C1.json")

    # load the json to dictionary
    json_bugs = load_json_to_dict(sample_path)

    # iterate over the json array
    for bug in tqdm(json_bugs, desc="Processing JSON Bugs"):
        # for bug in json_bugs:
        bug_title = html.unescape(bug['bug_title'])
        bug_description = html.unescape(bug['bug_description'])
        project = bug['project']
        sub_project = bug['sub_project']
        version = bug['version']
        es_results = bug['es_results']

        score_llm_results = llm_scoring(es_results, bug_title, bug_description, llm=llm, model_path=MODEL_PATH)

        bug['es_results'] = score_llm_results

    # Save output to Intelligent_Feedback directory
    json_save_path = str(script_dir / "Output" / "Intelligent_Feedback")
    JSON_File_IO.save_Dict_to_JSON(json_bugs, json_save_path, "Mistral_ZERO.json")
