1. 更新`devcontainer.json` 使用最新的nvidia-pytorch镜像 (不更新的话，安装第2步的包会出现版本错误)，这个image是和host的CUDA环境独立开的，不用担心环境配置问题。

```json
...
 "image": "nvcr.io/nvidia/pytorch:25.05-py3",
...
```

2. Open the VS Code terminal (which is inside the container):


```bash
pip install transformers accelerate bitsandbytes sentencepiece
```
3. Python脚本实践，如下所示，设置好`MOEDL_ID`、`os.environment['HF_ENDPONIT']`、`os.environment['HF_TOKEN']`，可以扁编写自己的python脚本了
```python
import torch
from transformers import pipeline, AutoTokenizer
import os

# --- Configuration ---
# Using the instruction-tuned version of Gemma 7B.
# You can also use "google/gemma-7b" for the base model.
MODEL_ID = "google/gemma-7b-it"

# --- Set Hugging Face Environment Variables for Mirror and Token ---
# Set the Hugging Face mirror endpoint
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Set your Hugging Face token here if you need to download models that require authorization.
# Replace "hf_YOUR_ACTUAL_TOKEN_HERE" with your actual Hugging Face token.
# You can generate a token from your Hugging Face settings: https://huggingface.co/settings/tokens
os.environ["HF_TOKEN"] = "使用自己的token" # Uncomment and replace with your token if needed for gated models [2, 6, 12]

# --- GPU Check ---
if not torch.cuda.is_available():
    print("CUDA is not available. Please check your Docker setup and GPU drivers.")
    exit()

num_gpus = torch.cuda.device_count()
print(f"Found {num_gpus} GPUs available to PyTorch.")
if num_gpus == 0:
    print("No GPUs detected by PyTorch. Exiting.")
    exit()

print(f"Your server has {num_gpus} RTX 3090s.")
print(f"Gemma 7B model in float16 precision requires ~14GB VRAM.")
print(f"Each RTX 3090 has 24GB VRAM.")

# --- Load Tokenizer and Model using Pipeline ---
print(f"\nLoading tokenizer and model for '{MODEL_ID}' using Hugging Face pipeline...")
print(f"Using device_map='auto' to distribute the model across available GPUs if beneficial.")
print(f"Using torch_dtype=torch.float16 for faster inference and reduced memory.")

try:
    # Load tokenizer separately to potentially apply chat template if needed
    # For Gemma, make sure you have a recent version of transformers (pip install -U transformers)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    generator = pipeline(
        "text-generation",
        model=MODEL_ID,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto", # Automatically distribute across GPUs
        # For some models, especially newer ones or those with custom code,
        # you might need trust_remote_code=True. Gemma usually doesn't.
        # trust_remote_code=True,
    )
    print("Model loaded successfully onto device(s).")

    # --- Inspect Model Distribution (Optional) ---
    if hasattr(generator.model, 'hf_device_map'):
        print("\nModel layer to device mapping (hf_device_map):")
        print(generator.model.hf_device_map)
    else:
        print("\nCould not retrieve hf_device_map from the pipeline's model.")
        if hasattr(generator.model, 'hf_device_map'): # Check again if it's on the inner model
             print(generator.model.model.hf_device_map)


    # --- Prepare Prompt for Gemma Instruct ---
    # Gemma instruct models have a specific chat format.
    # The pipeline should ideally handle this for simple prompts,
    # but for more complex interactions, using the chat template is better.
    user_prompt = "Write a short poem about the joy of discovery."

    # Option 1: Simple prompt (pipeline might auto-format)
    # prompt_to_send = user_prompt

    # Option 2: Explicitly use the chat template (more robust for instruct models)
    # This creates the structured prompt like:
    # <start_of_turn>user
    # Write a short poem about the joy of discovery.<end_of_turn>
    # <start_of_turn>model
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    prompt_to_send = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(f"\nGenerating text for prompt:\n'{prompt_to_send}'")

    # --- Perform Inference ---
    sequences = generator(
        prompt_to_send,
        max_new_tokens=150,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        eos_token_id=generator.tokenizer.eos_token_id # Important for stopping generation
    )

    print("\n--- Generated Text ---")
    for i, seq in enumerate(sequences):
        print(f"Result {i+1}:")
        # The pipeline output is a dictionary, and 'generated_text' contains the full text.
        # For Gemma with chat template, the output will include the prompt structure.
        # You might want to parse out just the model's response.
        full_text = seq['generated_text']
        print(full_text)

        # Attempt to extract only the model's response if chat template was used
        model_response_start = "<start_of_turn>model\n"
        if model_response_start in full_text:
            response_only = full_text.split(model_response_start, 1)[-1]
            print("\n--- Model's Response Only ---")
            print(response_only.strip())
        print("--------------------")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Troubleshooting tips:")
    print("- Ensure you have accepted Gemma's terms on its Hugging Face model page (if any).")
    print("- Ensure you are logged in via `huggingface-cli login` if facing auth issues (though Gemma is public).")
    print("- Check your internet connection inside the Docker container.")
    print("- Check available VRAM on your GPUs (use `nvidia-smi` in another terminal on the host).")
    print("- Ensure 'transformers' and 'accelerate' libraries are up-to-date.")

finally:
    print("\nScript finished.")
    # Clean up (optional)
    # if 'generator' in locals():
    #     del generator
    # if 'tokenizer' in locals():
    #     del tokenizer
    # torch.cuda.empty_cache()
```

