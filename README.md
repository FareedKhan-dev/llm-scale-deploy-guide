<!-- omit in toc -->
# Serving LLM for 100K Parallel Queries
**A Developer's Guide to Hosting 3B LLM for Millions of Queries in Parallel**
Whether you are building agents, RAG bots, or LLM apps, the core of your product is usually an LLM accessed via API. Providers like [Together.ai](https://www.together.ai/) optimize these models for efficient and scalable use, rather than hosting them in full precision.

These techniques include **weight quantization (e.g., W4A16, W4A8)**, **KV caching**, **speculative decoding**, and much more all used during inference. On the deployment side, Kubernetes and Docker enable parallel processing in the cloud. Every component is optimized to handle millions of queries efficiently.

In this blog, we are going to serve our own LLaMA model to handle around 102K parallel queries by experimenting with different optimization techniques to come up with proper solution.

> We will focus on optimizing the model architecture and cloud deployment, and then evaluate our optimized LLM using metrics, including latency, model memory consumption, and accuracy.

<!-- omit in toc -->
## Table of Contents
- [Our Development + Deployment Pipeline](#our-development--deployment-pipeline)
- [Setting up the Environment](#setting-up-the-environment)
- [Our Evaluation Datasets](#our-evaluation-datasets)
- [Evaluating Full Precision Model (Baseline)](#evaluating-full-precision-model-baseline)
- [Performing W4A16 Quantization](#performing-w4a16-quantization)
- [Comparing Base vs W4A16 vs W8A8](#comparing-base-vs-w4a16-vs-w8a8)
- [Is W4A8 Quantization a Better Option?](#is-w4a8-quantization-a-better-option)
- [Why Long Inputs is an Issue?](#why-long-inputs-is-an-issue)
- [Applying SDPA/SDPA PAGED for Long Inputs](#applying-sdpasdpa-paged-for-long-inputs)
- [Key-Value Cache Implementation](#key-value-cache-implementation)
- [Optimizing KV Cache for Multi-Round Conversations](#optimizing-kv-cache-for-multi-round-conversations)
- [Performing Prompt Lookup Decoding](#performing-prompt-lookup-decoding)
- [Speculative Decoding](#speculative-decoding)
- [Our Deployment Repo Architecture](#our-deployment-repo-architecture)
- [Creating Fast API Server](#creating-fast-api-server)
- [Containerizing with Docker](#containerizing-with-docker)
- [Deploying on Kubernetes](#deploying-on-kubernetes)
- [Creating GCP Cloud Cluster](#creating-gcp-cloud-cluster)
- [Launching the Service](#launching-the-service)
- [100K Parallel Queries Processing Test](#100k-parallel-queries-processing-test)
- [Summarizing Everything](#summarizing-everything)

## Our Development + Deployment Pipeline
Since we are experimenting with many trials and errors involving model weights, latency optimization, and deployment on Kubernetes clusters, it’s a good idea to visualize this pipeline first.

![Our pipeline](https://miro.medium.com/v2/resize:fit:1250/1*ViUdlgAoFVtl4409sKOdog.png)

We will be using two evaluation datasets, one for the experimental phase and another for evaluating the deployment phase. We will use three key evaluation metrics:

* **Latency**, which helps reduce LLM API call time during parallel processing
* **Accuracy**, which evaluates the quality of generated answers after applying each optimization algorithm
* **Peak memory consumption**, which tracks the average memory used during LLM inference on evaluation data

We will begin with **model weight optimization techniques** using trial and error. In this phase, we will apply different algorithms to optimize the model weights and evaluate each technique. After that, we will focus on **optimizing the inference steps** of the LLM to improve on-the-spot processing efficiency.

Finally, we will move on to **deployment phase** optimization using **Kubernetes** and **Docker**. Once that is complete, we will run evaluation using approximately 100,000 queries on the deployed system to measure how efficient it has become.

## Setting up the Environment
Before we begin testing different optimization techniques for both the development and deployment stages, we need to import the necessary libraries and create some utility functions that will be used throughout this blog.

Our target LLM is LLaMA version 3.2 with 3B parameters, and our goal is to serve it efficiently for 100,000 queries using parallel processing. So let’s define the model and import the required libraries.
```python
# Import necessary libraries from Hugging Face Transformers for model and tokenizer handling
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# Import PyTorch for tensor operations
import torch
# Import time for measuring performance
import time
# Import tqdm for progress bars in Jupyter notebooks
from tqdm.notebook import tqdm
# Import json for handling JSON data
import json

LLM_API_KEY = "YOUR_LLM_API_KEY"  # Replace with your actual API key (OpenAI, HuggingFace, Nebius, Together AI etc.)


# Defining the model ID (We are using Llama-3.2 1B model)
model_id = "meta-llama/Llama-3.2-1B"
```
We need an evaluator, and during our experimental phase, we will use the LLaMA-3 70B LLM as an evaluator for our pipeline.

It will assess the generated responses from our LLaMA 3.2 1B model after applying different algorithms and compare them to the ground truths to measure performance.

Let’s create a simple script that will evaluate the generated responses.
```python
import os, re
from openai import OpenAI

# Initialize Nebius OpenAI client
def setup_client(api_key=None):
    key = api_key or os.getenv("NEBIUS_API_KEY")
    return OpenAI(base_url="https://api.studio.nebius.com/v1/", api_key=key) if key else None

# Extract score from text using regex
def extract_score(text):
    for p in [r"(\d+\.?\d*)/1\.0", r"Score:\s*(\d+\.?\d*)", r"(\d+\.?\d*)"]:
        m = re.search(p, text)
        if m:
            s = float(m.group(1))
            return s if s <= 1 else s / 10 if s <= 10 else s / 100
    return 0.0

# Get similarity score between two answers
def get_score(client, gen, truth, model="deepseek-ai/DeepSeek-V3"):
    prompt = "You are a lenient evaluator. Score from 0.0 to 1.0. Only return the number."
    msg = f"[Generated Answer]:\n{gen}\n\n[Ground Truth Answer]:\n{truth}"
    res = client.chat.completions.create(model=model, messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": msg}
    ])
    return extract_score(res.choices[0].message.content.strip())
```
We have used a simple prompt template that scores the generated answer against the ground truth. On my GitHub, this script is more advanced and includes a detailed system prompt to make the LLM a stricter and more accurate evaluator.

We also need a function to get the memory usage of the LLM after it is loaded. Let’s create that function now.
```python
def get_model_memory_footprint(model):
    """
    Get the memory footprint of a model in megabytes.
    
    Args:
        model: The model to measure memory footprint for.
    
    Returns:
        float: Memory footprint in megabytes, rounded to 2 decimal places.
    """
    # Get memory footprint in bytes and convert to megabytes
    memory_bytes = model.get_memory_footprint()
    memory_mb = memory_bytes / (1024 * 1024)
    
    return round(memory_mb, 2)
```
And finally, we need a `generate_text` method or an LLM run function. This will help us avoid duplicating code, since we will be running many experiments. So let’s create that function as well.
```python
# Function to generate text from a model and tokenizer with optional memory and time measurement
def generate_text(tokenizer, model, **kwargs):
    """
    Generate text using a given tokenizer and model.
    
    Args:
        tokenizer: The tokenizer for encoding input text.
        model: The language model for text generation.
        **kwargs: Additional arguments for model.generate(), including:
            - input_text (str): The prompt to generate from.
            - max_new_tokens, do_sample, temperature, top_p, top_k, pad_token_id, etc.
    
    Returns:
        tuple: A tuple containing the generated text (str), peak memory usage in MB (float or None),
               and generation time in seconds (float).
    """
    # Extract the input text from keyword arguments
    input_text = kwargs.pop('input_text')
    # Tokenize the input text and convert it to PyTorch tensors
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Check if a CUDA-enabled GPU is available
    if torch.cuda.is_available():
        # Reset peak memory statistics for the current CUDA device
        torch.cuda.reset_peak_memory_stats()
        # Get the device of the model (e.g., 'cuda:0')
        device = next(model.parameters()).device
        # Move the input tensors to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Record the start time for generation
    start_time = time.time()
    # Generate text using the model with the provided inputs and generation arguments
    output_tokens = model.generate(**inputs, **kwargs)
    # Record the end time for generation
    end_time = time.time()
    
    # Calculate the total generation time and round it to 3 decimal places
    generation_time = round(end_time - start_time, 3)
    
    # Check if a CUDA-enabled GPU is available to measure memory
    if torch.cuda.is_available():
        # Measure the peak memory allocated on the GPU during generation
        # Convert bytes to megabytes (MB) and round to 3 decimal places
        peak_memory = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 3)
    else:
        # If no GPU is available, set peak memory to None
        peak_memory = None
        
    # Decode the generated tokens back into a string, skipping special tokens
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    # Return the generated text, peak memory usage, and generation time
    return generated_text, peak_memory, generation_time
```
This function will return the generated text, the computation time (latency), and the memory used during inference.

## Our Evaluation Datasets
We will be using two datasets. One is for experimenting, which was created using ChatGPT and contains 50 queries.

The other dataset is from Hugging Face, provided by [Microsoft MS MARCO](https://huggingface.co/datasets/microsoft/ms_marco), containing more than 102,000 queries. We will use it to evaluate how well our constructed pipeline performs at scale.

Let’s load the experimental data and print its output.
```bash
# Import the json library to work with JSON files
import json

# --- Load Evaluation Dataset ---
# Open the evaluation data file in read mode
with open('eval_data.json', 'r') as file:
    # Load the JSON content from the file into the 'eval_data' variable
    eval_data = json.load(file)

# --- Display Sample Questions and Answers ---
# Print the first two questions and their answers from the dataset to verify it's loaded correctly
print("Sample Question 1:", eval_data[0]['q'])
print("Sample Answer 1:", eval_data[0]['a'])
print("\nSample Question 2:", eval_data[1]['q'])
print("Sample Answer 2:", eval_data[1]['a'])

# --- Total Number of Questions ---
# Print the total number of questions in the evaluation dataset
print("Total Number of Questions:", len(eval_data))

### OUTPUT ####
Sample Question 1: What color is the sky?
Sample Answer 1: The sky is blue.

Sample Question 2: How many legs does a cat have?
Sample Answer 2: It has four legs.
Total Number of Questions: 49
```
Similarly, we can load and print a sample of the [MS MARCO dataset](https://huggingface.co/datasets/microsoft/ms_marco), which will be used for the final evaluation.
```python
from datasets import load_dataset

# Load MS MARCO (v1.1 subset, 'train' split)
dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")

# Print the first 5 queries
for i in range(5):
    print(f"Query {i+1}: {dataset[i]['query']}")


#### OUTPUT ####
Query 1: what is rba
Query 2: was ronald reagan a democrat
Query 3: when did the iphone 6 come out
Query 4: how many calories in an avocado
Query 5: define self respect
```
Great, so now that we have defined the functions and initialized the datasets, we can start performing experiments, beginning with creating the baseline, which involves running inference using the original model.

## Evaluating Full Precision Model (Baseline)
We need to evaluate our baseline model by running inference on the original setup. This will obviously consume GPU cloud infrastructure resources, especially if many parallel queries are processed, so it is not ideal for regular use.

However, we still need to create this baseline to measure latency, accuracy, and peak memory usage. Let’s go ahead and perform that.
```python
# Load the tokenizer for the specified model
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the causal language model with bfloat16 precision on CPU
original_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto" # Use "auto" for automatic device mapping if you have a GPU
)
```
So we have initialized our base model. Let’s run inference on our evaluation data using the predefined modules, and then we can evaluate the outputs using the LLM-as-a-judge process. Let’s do that.
```python
# Running the evaluation on the model with the evaluation dataset
base_model_results = evaluate_model(original_model, tokenizer, eval_data)

# --- Evaluate with LLM Judge ---
# Use the 'evaluate_with_judge' function to assess the model's performance.
# This function takes the model's results and an API key for the judging service.
# It returns a DataFrame with detsailed results and a dictionary of overall metrics.
# Note: You need to replace "your_nebius_api_key_here" with your actual Nebius API key.
base_model_results_df, base_model_metrics = evaluate_with_judge(
    base_model_results,  # The results from your model evaluation
    api_key=LLM_API_KEY,  # API key for the judging service
    model_name="meta-llama/Llama-3.3-70B-Instruct"  # Name of the model being evaluated
)
```
So now that we have performed two evaluations, first generating the text and then evaluating the generated responses against the ground truth, let’s print the evaluation metrics.
```python
# --- Display Base Model Metrics ---
# Print the overall performance metrics for the base model.
# These metrics include average latency, memory usage, and similarity score.
print("Base Model Performance Metrics:")
for key, value in base_model_metrics.items():
    # Print each metric with its corresponding value, formatted to 4 decimal places
    print(f"- {key.replace('_', ' ').title()}: {value:.4f}")


#### OUTPUT ####
Base Model Performance Metrics:
- Avg Latency: 4.4598
- Avg Memory: 1240.2071
- Avg Score: 0.3939
```
Ah, so the base average latency is around 4 seconds per query, with 1240 MB memory usage per question, and a performance score of 40 percent. Since we are asking generic questions, an LLM of 1B size might fail often.

This now serves as our baseline. If any new algorithm performs better, we will use this baseline for comparison to see whether the new approach outperforms it or not. So let’s get started.

## Performing W4A16 Quantization
Weight-only quantization is the first step in optimizing the memory of LLMs because they are composed of tensors which are weights and their precision can be reduced through a process called quantization.

In weight-only quantization, we quantize the model weights except for the activation layers which remain at higher or original precision as they are responsible for generating text.

![Weight only Quantization](https://miro.medium.com/v2/resize:fit:875/1*Dn-OBWk8gExvEWGwFCTLqA.png)

Here are a few different ways to perform this quantization.

| Model | Avg Latency (s) | Avg Memory (MB) | Avg Score |
| :--- | :--- | :--- | :--- |
| **Base Model** | 4.4598 | 1240.2071 | 0.3939 |
| **W4A16** | 1.4541 | 1053.8099 | 0.4224 |
| **W8A8** | 4.2488 | 1489.3576 | 0.4245 |
| **W4A8** | 3.2488 | 952.3250 | 0.3677 |

We are going to test the W4A16 approach which means the model weights are quantized to 4 bits while the activation layer weights remain at a higher precision of 16 bits.

So let’s define the quantization configuration required to load the model in the W4A16 format.
```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Define quantization config: 4-bit weights (W4A16)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,          # optional: improves accuracy
    bnb_4bit_quant_type="nf4",               # "nf4" (normal float 4-bit) or "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16    # use bfloat16 for activations (W4A16)
)
```
The `bnb_4bit_compute_dtype` sets the activation layers to the original or higher precision while the parameter `load_in_4bit = True` enables 4-bit quantization for the model weights.

Now we can easily load this W4A16 model and start performing evaluation on our evaluation data.
```python
# Load the model with 4-bit quantization (weights only)
w4a16_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"  # use "auto" for best device placement (GPU/CPU)
)
```
Let’s check the footprint of our quantized model to see how much memory has been reduced.
```python
# Memory footprint of the quantized model
w4a16_model_memory = get_model_memory_footprint(w4a16_model)
print("Quantized Model Memory Footprint (MB):", w4a16_model_memory)


#### OUTPUT ####
Quantized Model Memory Footprint (MB): 965.13
```
Our quantized model memory has been reduced to nearly half the size from 2 GB to 0.9 GB. In terms of memory, this is a significant reduction and there is a strong possibility that latency will also improve.

So let’s evaluate this model and see how its performance compares to the original base model.
```python
# --- Evaluate the W4A16 Quantized Model ---
# Running the evaluation on the quantized model with the evaluation dataset
w4a16_model_results = evaluate_model(w4a16_model, tokenizer, eval_data)

# --- Evaluate with LLM Judge ---
# Use the 'evaluate_with_judge' function to assess the quantized model's performance.
w4a16_model_results_df, w4a16_model_metrics = evaluate_with_judge(
    w4a16_model_results,  # The results from your quantized model evaluation
    api_key=LLM_API_KEY,  # API key for the judging service
    model_name="meta-llama/Llama-3.3-70B-Instruct"  # Name of the model being evaluated
)
```
Now that we have performed the evaluation, let’s observe the results and see what metric values it has produced.
```python
# --- Display W4A16 Quantized Model Metrics ---
# These metrics include average latency, memory usage, and similarity score.
print("W4A16 Quantized Model Performance Metrics:")
for key, value in w4a16_model_metrics.items():
    # Print each metric with its corresponding value, formatted to 4 decimal places
    print(f"- {key.replace('_', ' ').title()}: {value:.4f}")


#### OUTPUT ####
W4A16 Quantized Model Performance Metrics:
- Avg Latency: 1.4541
- Avg Inference Memory Consumption: 1053.8099
- Avg Score: 0.4224
```
The results are above our expectations because we are not quantizing the activation layer weights which are responsible for generating text. You might think that latency would remain the same, but by quantizing the other model weights, the memory consumption per query has also been reduced.

A key advantage is that the score is very close to our original base model evaluation result, which is a good sign.

But the question is whether we can go further, such as using W8A8.
 So let’s perform that next.

## Comparing Base vs W4A16 vs W8A8
While W4A16 uses 16-bit precision for activation layers and 4-bit precision for other weights, in W8A8 all weights remain in 8-bit precision.

![W8A8 Optimization Approach](https://miro.medium.com/v2/resize:fit:875/1*652fAIs_E8svi5EUUYwDWQ.png)

Just like we performed the evaluation of the previous approach, let’s now evaluate W8A8 in the same way. We first need to initialize the configuration of the base model that will convert the model into W8A8.
```python
# Configure for 8-bit weight + activation quantization (W8A8)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,                           # W8
    llm_int8_threshold=6.0,                      # Optional: threshold for outlier detection
    llm_int8_has_fp16_weight=False,              # Force 8-bit only mode (no fallback to fp16)
    llm_int8_enable_fp32_cpu_offload=True,       # Optional: offload to CPU for better memory management
)

# defining the tokenizer again for clarity
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model with 8-bit weights and 8-bit activations
w8a8_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"  # Will use GPU if available
)
```
Let’s check the model footprint first and see how much memory it uses.
```python
# Memory footprint of the quantized model
w8a8_model_memory = get_model_memory_footprint(w8a8_model)
print("Quantized Model Memory Footprint (MB):", w8a8_model_memory)


#### OUTPUT ####
Quantized Model Memory Footprint (MB): 1429.13
```
The memory size has been reduced, but since all the weights are quantized to 8-bit, it reduces memory usage but not as significantly as W4A16. Now let’s perform the evaluation.
```python
# --- Evaluate the W8a8 Quantized Model ---
# Running the evaluation on the quantized model with the evaluation dataset
w8a8_model_results = evaluate_model(w8a8_model, tokenizer, eval_data)


# --- Evaluate with LLM Judge ---
# Use the 'evaluate_with_judge' function to assess the quantized model's performance.
w8a8_model_results_df, w4a16_model_metrics = evaluate_with_judge(
    w8a8_model_results,  # The results from your quantized model evaluation
    api_key=LLM_API_KEY,  # API key for the judging service
    model_name="meta-llama/Llama-3.3-70B-Instruct"  # Name of the model being evaluated
)


#### OUTPUT ####
W8A8 Quantized Model Performance Metrics:
- Avg Latency: 4.2488
- Avg Inference Memory Consumption: 1489.3576
- Avg Score: 0.4245
```
Now that we have computed the evaluation results, they are quite similar to the base model in terms of latency and accuracy. However, it’s better to visualize these results to clearly see which quantization approach performs better overall.

![Weight only quantization comparison](https://miro.medium.com/v2/resize:fit:875/1*TJFuL0jsGf6F9_hfPpSC8Q.png)

if we look at the overall view, W4A16 is the approach that works best for us because its latency, along with sustained accuracy compared to the base model, is much better.

We will explore this further when we perform batch processing later in this blog. However, there is still one more quantization approach remaining, so let’s look at that next.

## Is W4A8 Quantization a Better Option?
There is still one more option, W4A8, which uses 4-bit weights and 8-bit activations. This configuration is important to consider when dealing with limited resources or when handling millions of parallel requests.

However, we first want to test the impact of this quantization on a simple evaluation to see whether it performs well compared to W4A16 based on a single query.

So let’s define its configuration and perform the evaluation.
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,               # Load weights in 4-bit (W4)
    bnb_4bit_use_double_quant=True, # Optional but recommended for better accuracy
    bnb_4bit_quant_type="nf4",      # NormalFloat4 quantization, good balance between speed and accuracy
    llm_int8_threshold=6.0,          # Optional threshold (can be adjusted or removed)
    llm_int8_has_fp16_weight=False, # Force 4-bit only mode
    llm_int8_enable_fp32_cpu_offload=True, # Optional CPU offload for memory management
)

# # Load the model with 4-bit weights and 8-bit activations
w4a8_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

# 
w4a8_model_memory = get_model_memory_footprint(w4a8_model)
print("W4A8 Quantized Model Memory Footprint (MB):", w4a8_model_memory)


#### OUTPUT ####
Quantized Model Memory Footprint (MB): 753.63
```
The memory usage of the model is lower than W4A16, which is a good sign. Now let’s perform the evaluation.
```python
# Evaluate W4A8 model results with the judge
w4a8_model_results_df, w4a8_model_metrics = evaluate_with_judge(
    w4a8_model_results,  # results from your W4A8 quantized model evaluation
    api_key=LLM_API_KEY,
    model_name="meta-llama/Llama-3.3-70B-Instruct"
)

# --- Evaluate with LLM Judge ---
# Use the 'evaluate_with_judge' function to assess the quantized model's performance.
w4a8_model_results_df, w4a16_model_metrics = evaluate_with_judge(
    w4a8_model_results_df,  # The results from your quantized model evaluation
    api_key=LLM_API_KEY,  # API key for the judging service
    model_name="meta-llama/Llama-3.3-70B-Instruct"  # Name of the model being evaluated
)

#### OUTPUT ####
W4A8 Quantized Model Performance Metrics:
- Avg Latency: 3.2488
- Avg Inference Memory Consumption: 952.325
- Avg Score: 0.3677
```
The performance has slightly decreased compared to the other approaches. Let’s visualize the results to better understand the differences.

![4 models weight only comparison](https://miro.medium.com/v2/resize:fit:875/1*2ExetcO_MuM4pKcfuFTkkA.png)

From the above experiments, it is clear that quantizing the activation layers to lower precision is not a good option in our case, as it affects both latency and accuracy.

Now we need to explore how we can optimize LLM memory and inference performance without using quantization.

## Why Long Inputs is an Issue?
Handling larger inputs for your LLM is a separate challenge, especially because our LLaMA 3.2 has a context window of nearly 128K tokens. When users fully utilize this context, it significantly increases GPU memory usage and introduces latency delays.

Let’s pass an [entire book (*Pride and Prejudice*)](https://giove.isti.cnr.it/demo/eread/Libri/joy/Pride.pdf) into the context of our LLM and ask a simple question to observe its performance impact.

First, let’s load the book and print the total number of words it contains.
```python
import fitz  # PyMuPDF

# Load the PDF
pdf_path = "story_book_chunked.pdf"
doc = fitz.open(pdf_path)

# Extract all text from the PDF
pdf_text = ""
for page in doc:
    pdf_text += page.get_text()

# Count total words in the PDF
word_count = len(pdf_text.split())

print("Total Words in PDF Book:", word_count)


#### OUTPUT ####
Total Words in PDF Book: 75357
```
The total number of words in the book is around 75K, which translates to approximately 90K to 100K tokens. Let’s pass the entire book into the memory of our W4A16 LLM and see whether it creates any performance or memory issues.
```python
# Define full prompt using the book content
pdf_prompt = pdf_text + "\n\nSummarize the main theme of this book."

# Run inference using the quantized model (e.g., w4a16)
generated_text, peak_memory, latency = generate_text(
    tokenizer,
    w4a16_model,
    input_text=pdf_prompt,
    max_new_tokens=200,
    do_sample=False,
    temperature=0.7,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

# Display Results
print("==== PDF Book Inference ====")
print("Latency:", latency)
print("Peak Memory Used (MB):", peak_memory)
print("Generated Output:\n", generated_text[-800:])


#### OUTPUT ####
Total Words in PDF Book: 75357

==== PDF Book Inference ====
Latency: 34.239 sec
Peak Memory Used (MB): 8532.74
Generated Output:
... human connection, and transformation through adversity.
```
The computation is expensive. I ran it on an L4 GPU from the Lightning AI platform. Even though we use our best W4A16 quantized model, the score is not important to calculate. Even five runs can heat up your GPU memory a lot. Imagine handling thousands of requests like this. It would definitely cost you a lot, even if you are doing parallel processing.

The problem exists in self-attention because it has query, key, and value matrices which are derived from the input sequence. Consider this self-attention formula.

![Attention Equation](https://miro.medium.com/v2/resize:fit:875/1*qcJkxPAIsHdNBAN5wWyGsg.png)

The problem is that since these Q, K, and V are derived from input sequences, they grow exponentially as the chat goes deeper.

This is obviously going to happen when you create an LLM-based app that depends on higher context. Over time, the chat context increases.

We’ll compute the memory needed to store **only the attention matrix** QKᵀ ** ** across multiple heads.

* N = sequence length
* H = number of attention heads = 40
* Using **bfloat16** precision = 2 bytes per value
* Total memory for all heads:

![Memory Formula](https://miro.medium.com/v2/resize:fit:875/1*9I1mieOVienpxcKgEF8UOA.png)

When using different sequence lengths this is what we get

| Sequence Length (N) | Attention Matrix Size (N x N) | Memory for One Head (MB) | Total Memory for 40 Heads (GB) |
| :--- | :--- | :--- | :--- |
| 1,024 | 1,048,576 | 2.0 | 0.08 |
| 4,096 | 16,777,216 | 32.0 | 1.28 |
| 8,192 | 67,108,864 | 128.0 | 5.12 |
| 16,384 | 268,435,456 | 512.0 | 20.48 |
| 32,768 | 1,073,741,824 | 2,048.0 | 81.92 |
| 128,000 | 16,384,000,000 | 31,250.0 | 1,250.0 |

Having only 40 attention heads and sequences of different lengths, you can see that even at 16K sequence tokens, it requires 19GB of memory for one run. This means it can easily consume an A100 GPU for a minimal number of queries when running in parallel.

To handle long input from users, we need to address this with a proper algorithm.

## Applying SDPA/SDPA PAGED for Long Inputs
SDPA (Scaled Dot Product Attention) is now being used to solve the long sequence issues during LLM inference.

It is now the core operation of the Transformer block, which has become the backbone of many language models and generative models.

The SDPA algorithm in short does this:

![SDPA Algorithm Workflow](https://miro.medium.com/v2/resize:fit:875/1*CPEWK6pM9pN3BBeac5Dglw.png)

* Avoids materializing the full QKᵀ matrix
* Uses fused kernels on GPU
* Streams computation to reduce memory

Let’s perform a simple example to see the performance difference before we apply it to our W4A16 LLM.
```python
import torch.nn.functional as F

# Define model parameters
seq_len = 8192
batch_size = 1
hidden_dim = 128
num_heads = 8
head_dim = hidden_dim // num_heads
device = "cuda"

# Create query, key, and value tensors
q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
k = torch.randn_like(q)
v = torch.randn_like(q)


# --- Manual Scaled Dot Product Attention (not memory efficient) ---
# 1. Calculate attention scores (Q @ K^T)
scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
# 2. Apply softmax to get attention weights
attn = torch.softmax(scores, dim=-1)
# 3. Multiply weights by values (Attn @ V)
output = torch.matmul(attn, v)

# --- Scaled Dot Product Attention (SDPA) ---
# Use PyTorch's built-in, memory-efficient scaled dot product attention
output = F.scaled_dot_product_attention(q, k, v, is_causal=True)


# Print output shape and peak CUDA memory usage
print("Output shape:", output.shape)
print(f"Peak CUDA Memory (SDPA): {torch.cuda.max_memory_allocated(device) / (1024**2):.2f} MB")
print(f"Peak CUDA Memory (Standard): {torch.cuda.max_memory_allocated(device) / (1024**2):.2f} MB")
```
So we are basically performing manual attention as it appears in the Transformer block in comparison to the SDPA attention approach and observing how much it affects the performance. Let’s see the output.
```python
#### OUTPUT ####
Output shape: torch.Size([1, 8, 8192, 16])
Peak CUDA Memory (SDPA): 612.42 MB
Peak CUDA Memory (Standard): 5372.35 MB
```
We chose the 8192 input sequence, which is a standard input token length for most LLMs. The memory difference is clear here, showing that SDPA is a clear winner in our case.

Let’s see what other algorithms are available besides SDPA.
```python
# To see which attention implementations are available in your transformers installation,
# you can inspect the `ALL_ATTENTION_FUNCTIONS` dictionary.
# This is useful for knowing what strings you can pass to the `attn_implementation` argument.
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

list(ALL_ATTENTION_FUNCTIONS.keys())


#### OUTPUT ####
['flash_attention_3',
 'flash_attention_2',
 'flex_attention',
 'paged_attention',
 'sdpa',
 'sdpa_paged',
 'eager_paged']
```
We have variations of SDPA and there is a lot to explore. We can simply perform an evaluation comparing SDPA with one of its variations, SDPA Paged, and see how it performs.

SDPA Paged divides the sequence into fixed-size pages (blocks) and computes attention in paged chunks, which is more beneficial for even longer sequences.

It’s time to run both SDPA and SDPA Paged algorithms on our quantized W4A16 LLM and see whether it has become more efficient or not.
```python
# SDPA (Sparse Distributed Parallel Attention) is a memory-efficient attention implementation.
w4a16_model_sdpa = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",  # use "auto" for best device placement (GPU/CPU)
    attn_implementation="sdpa",  # Use memory-efficient SDPA,
)

# SDPA_PAGED (Sparse Distributed Parallel Attention with Paged Memory) is another memory-efficient attention implementation.
w4a16_model_sdpa_paged = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",  # use "auto" for best device placement (GPU/CPU)
    attn_implementation="sdpa_paged",  # Use memory-efficient SDPA_PAGED,
)
```
Let’s evaluate them both and see their results.
```python
# Evaluate the W4A16 SDPA Model and W4A16 SDPA Paged Model
w4a16_model_sdpa_results = evaluate_model(w4a16_model_sdpa, tokenizer, eval_data)
w4a16_model_sdpa_paged_results = evaluate_model(w4a16_model_sdpa_paged, tokenizer, eval_data)

# --- Evaluate both results with LLM Judge ---
w4a16_model_sdpa_results_df, w4a16_model_sdpa_metrics = evaluate_with_judge(
    w4a16_model_sdpa_results,  # The results from your SDPA model evaluation
    api_key=LLM_API_KEY,  # API key for the judging service
    model_name="meta-llama/Llama-3.3-70B-Instruct"  # Name of the model being evaluated
)
w4a16_model_sdpa_paged_results_df, w4a16_model_sdpa_paged_metrics = evaluate_with_judge(
    w4a16_model_sdpa_paged_results,  # The results from your SDPA Paged model evaluation
    api_key=LLM_API_KEY,  # API key for the judging service
    model_name="meta-llama/Llama-3.3-70B-Instruct"  # Name of the model being evaluated
)

# Printing Metrices values of both (SDPA, SDPA PAGED)
Print(w4a16_model_sdpa_metrics, w4a16_model_sdpa_paged_metrics)
```
```
SDPA Performance Metrics:
- Avg Latency: 1.4087
- Avg Inference Memory Consumption: 1056.7745
- Avg Score: 0.4286

SDPA PAGED Performance Metrics:
- Avg Latency: 2.1583
- Avg Inference Memory Consumption: 1060.4898
- Avg Score: 0.4286
```
The paged version is not performing well in our case, while the simple SDPA is much more effective, so we will stick with that.

Let’s visually see how impactful SDPA can be based on increasing input sequences.

![Memory Consumptions on different length sequences](https://miro.medium.com/v2/resize:fit:1250/1*VKwotibrlZBiMTU6KVctog.png)

For different numbers of heads, the computation is extremely high and increases exponentially based on sequence length. So it is very important to implement the SDPA attention algorithm into your LLM.

This allows users to enjoy complex and lengthy context from your model without waiting too long.

## Key-Value Cache Implementation
Optimizing the model’s weights is one way to make an LLM efficient in a production environment, but we can also optimize the inference step when users are running instant queries to reduce latency and avoid making them wait for several seconds.

Let’s quickly understand KV Cache. Imagine having a conversation where, every time the other person replies, they need to go back and re-read everything you’ve said so far just to know what to say next. It would take time and effort, definitely not the most efficient way to talk.

Transformer models work in a similar way. To generate the 1000th token, the model has to process the previous 999 tokens.

This becomes especially important if you are building a RAG bot, where chunked data is passed as context. That context can be huge, and for a new conversation that relies on the same context, this can create a significant cost issue.

> Caching should only be used for inference. It may cause unexpected errors if it’s enabled during training.

![Attention Mechanism Formula](https://miro.medium.com/v2/resize:fit:875/1*6Xf2HpLpsFhOAASFklLX8g.png)

In transformer models, **scaled dot-product attention** is used to let tokens **“pay attention”** to others. During **inference**, we process one token at a time, so we only need:

* The **latest query** `q_t`
* All **past keys and values** `[k₁...kₜ₋₁], [v₁...vₜ₋₁]` (cached)

Instead of recomputing everything, we **cache**:

* `K_cache ← concat(K_past, k_t)`
* `V_cache ← concat(V_past, v_t)`

This caching happens **per layer**, saving time and memory by avoiding repeated computation for past tokens.

| Without KV Cache | With KV Cache |
| :--- | :--- |
| Re-computes past tokens | Reuses past computations |

Let’s look at the available KV cache techniques in the Transformer module:
```python
import transformers.cache_utils as cache_utils

# List all cache classes available in the module
cache_classes = [cls for cls in dir(cache_utils) if "Cache" in cls and not cls.startswith("_")]
print("Available cache strategies:")
for cls in cache_classes:
    print(f"- {cls}")


#### OUTPUT ####
Available cache strategies:
- Cache
- CacheConfig
- DynamicCache
- EncoderDecoderCache
- HQQQuantizedCache
- HybridCache
- HybridChunkedCache
- MambaCache
- OffloadedCache
- OffloadedHybridCache
- OffloadedStaticCache
- QuantizedCache
- QuantizedCacheConfig
- QuantoQuantizedCache
- SinkCache
- SlidingWindowCache
- StaticCache
- StaticCacheConfig
```
There are many KV cache algorithms available. We can focus on the most commonly used techniques such as static, dynamic and others.

Here is a quick summary of what most famous algorithm means.

* **DynamicCache:** Best for variable-length prompts and chat; flexible and widely used.
* **StaticCache:** Fast and efficient for fixed-length inputs or benchmarking.
* **SlidingWindowCache:** Maintains recent context only; ideal for streaming and limited memory.
* **OffloadedCache:** Enables long-context or large models on low-VRAM devices by using CPU/disk.
* **QuantizedCache:** Reduces memory usage by storing cache in lower-precision formats.

Let’s define these top performing techniques in a list:
```python
# Define a list of KV cache strategies to evaluate
# These are different implementations for caching key-value pairs during text generation
kv_cache_strategies = [
    "dynamic", # DynamicCache: dynamically manages cache size based on input length
    "static", # StaticCache: uses a fixed-size cache for all inputs
    # "sliding_window", # SlidingWindowCache: maintains a sliding window of the most recent tokens
    "quantized", # QuantizedCache: uses quantization techniques to reduce memory usage
    # The following might require specific model architectures or additional setup
    # "offloaded_static",
    # "hybrid",
    # "mamba",
]
```
We can simply loop through all algorithms one by one and perform the evaluation.
```python
# Perform kv cache evaluation on w4a16_model_sdpa (W4A16 SDPA model)
for cache_strategy in kv_cache_strategies:
    print(f"Evaluating with cache strategy: {cache_strategy}")
    
    # Evaluate the model with the specified cache strategy
    w4a16_model_sdpa_kv_results = evaluate_model(
        w4a16_model_sdpa,
        tokenizer,
        eval_data,
        use_cache=True,  # Enable KV caching
        cache_implementation=cache_strategy,  # Use specified cache strategy
        pad_token_id=tokenizer.eos_token_id,  # Ensure padding token is set
    )
    
    # --- Evaluate with LLM Judge ---
    w4a16_model_sdpa_kv_results_df, w4a16_model_sdpa_kv_metrics = evaluate_with_judge(
        w4a16_model_sdpa_kv_results,  # The results from your model evaluation with KV cache
        api_key=LLM_API_KEY,  # API key for the judging service
        model_name="meta-llama/Llama-3.3-70B-Instruct"  # Name of the model being evaluated
    )
    
    # Print the evaluation results for the current cache strategy
    print(f"{cache_strategy} Performance Metrics:")
    for key, value in w4a16_model_sdpa_kv_metrics.items():
        print(f"- {key.replace('_', ' ').title()}: {value:.4f}")
```
We evaluate the model on three different KV cache strategies and visualize the performance.
```
Dynamic Performance Metrics:
- Avg Latency: 1.2086
- Avg Inference Memory Consumption: 1056.7745
- Avg Score: 0.4020

Static Performance Metrics:
- Avg Latency: 1.2291
- Avg Inference Memory Consumption: 1056.7745
- Avg Score: 0.4020

Quantized Performance Metrics:
- Avg Latency: 1.2588
- Avg Inference Memory Consumption: 1053.8099
- Avg Score: 0.4020
```
There is a clear performance difference as the latency has decreased while the score remains the same which is a good sign. Let’s visualize this result for better interpretation.

![W4A16_SPDA_KV test](https://miro.medium.com/v2/resize:fit:875/1*Mu_vdSPMiUsm_0kKiE38Kg.png)

Dynamic is working best in our case in terms of latency while the score and memory consumption have not been reduced compared to previous steps.

This implementation shows how we can use KV cache to improve the latency of our efficient quantized model

## Optimizing KV Cache for Multi-Round Conversations
We haven’t gone over the multi-turn conversation which is a capability every AI app has to store a certain number of previous messages.

This is highly important when dealing with the KV cache as we did earlier. Let’s look at this example:
```python
Example of Multi Chat Conversation

User: What's the capital of Japan?
Assistant: Tokyo is the capital of Japan.
User: And what about China?
```
The model generates replies one at a time using **auto-regressive decoding**.

**First reply:**

* The cache is empty.
* The model reads: `"User: What's the capital of Japan?"`
* It generates: `"Tokyo is the capital of Japan."`
* While doing this, it stores useful data (keys and values) in memory.

**Second reply:**

* The model sees this full history: `"User: What's the capital of Japan? \n Assistant: Tokyo is the capital of Japan. \n User: And what about China?"`
* But thanks to the cache, it doesn’t have to reprocess the whole thing.
* It reuses the stored data and only processes: `"User: And what about China?"`
* Then it generates: `"Beijing is the capital of China"`.

Two important things should be noted here:

* **Context matters:** When the user asks *“And what about China?”*, the model understands it’s about capitals because of the earlier question.
* **Cache saves work:** The key-value cache lets the model reuse earlier parts of the chat without re-reading everything, making it faster and more efficient.

Let’s visualize how KV cache works in multi-turn chat.

![Multi Chat conversation KV Cache](https://miro.medium.com/v2/resize:fit:1250/1*d9SvjMcEKdrtgfmN5Z9JVQ.png)

Now let’s implement this logic in code. In the very first turn the cache is empty with nothing to store so we ask a question and the model generates an answer.
```python
# First question
question = "What is the capital of Japan?"
model_inputs = tokenizer(question, return_tensors='pt')

# Generate the first answer
output = w4a16_model_sdpa.generate(
    **model_inputs,
    max_new_tokens=30,
    return_dict_in_generate=True,
    use_cache=True,  # Enable KV caching
    cache_implementation="dynamic",  # Use best cache strategy for ourcase
)

# Detokenizing the answer
answer1 = tokenizer.batch_decode(output.sequences)[0]
print(answer1)


#### OUTPUT ####
The capital of Japan is Tokyo.
```
But then we also need to store the decoded values for future conversation so that the model can use them in the next turns.
```python
# Follow-up question, using past_key_values for efficiency
follow_up = "What about China?"
model_inputs = tokenizer(answer1 + "\n" + follow_up, return_tensors='pt')
```
The key parameter which is very important to keep in mind is `past_key_values` which prevents the model from recalculating the KV cache of the previous conversation that has already been processed.
```python
output = w4a16_model_sdpa.generate(
    **model_inputs,
    past_key_values=output.past_key_values, # It let's model avoid recalc of KV Cache
    max_new_tokens=30,
    return_dict_in_generate=True,
    use_cache=True,  # Enable KV caching
    cache_implementation="dynamic",  # Use best cache strategy for ourcase
)
answer2 = tokenizer.batch_decode(output.sequences)[0][len(answer1 + "\n" + follow_up):]
print(answer2)


#### OUTPUT #####
The capital of China is Beijing.
```
Great so now that we have learned the logic of optimizing multi-round conversation during inference using the KV cache strategy we can later apply that logic when deploying the model by simply checking whether the conversation is the first turn or not and then using the `past_key_values` parameter accordingly

## Performing Prompt Lookup Decoding
In many cases your hosted LLM needs the original tokens that are already available in the context of the LLM.

These tasks include summarization text simplification or question answering. Smaller LLMs are often part of a larger architecture which handles such tasks to reduce the overall workload.

![Reuse of Tokens](https://miro.medium.com/v2/resize:fit:875/1*Xo4LD9S8N8tj1qXbPb4E2Q.png)

Prompt lookup is a technique that lets the model refer to the tokens in the input to avoid recalculating them which helps save more latency for the LLM.
```python
# First question
# Define the prompt for the model
question = "Name the capital city of Japan?"
# Tokenize the input question and move it to the GPU for processing
model_inputs = tokenizer(question, return_tensors='pt').to("cuda")


# Generate a response using the model with prompt lookup decoding
output = w4a16_model_sdpa.generate(
    **model_inputs,
    max_new_tokens=5, # Limit the generation to 5 new tokens
    return_dict_in_generate=True, # Return a detailed output object
    use_cache=True,  # Enable KV caching to speed up decoding
    
    # Specify the number of prompt tokens to use for lookup decoding.
    # This speeds up processing by matching the initial tokens of the prompt
    # against a pre-computed table, avoiding redundant computations.
    prompt_lookup_num_tokens=3,

)
# Decode the generated tokens back into a human-readable string
answer = tokenizer.batch_decode(output.sequences)[0]
# Print the final answer
print(answer)


#### OUTPUT ####
<|begin_of_text|>Name the capital city of Japan?  Answer: Tokyo
```
The `prompt_lookup_num_tokens` parameter is the key here as it decides how much overlap you need to avoid recalculating those tokens. A value of 3 for overlap means phrases like **"is the capital"** can be reused. Let's evaluate this performance with our KV cache approach.
```graphql
# Performing evaluation with prompt lookup decoding with kv cache
prompt_lookup_results = evaluate_model(
    w4a16_model_sdpa,  # The model to evaluate
    tokenizer,         # The tokenizer for encoding input text
    eval_data,         # The evaluation dataset
    use_cache=True,    # Enable KV caching
    cache_implementation="dynamic",  # Use the dynamic cache implementation
    prompt_lookup_num_tokens=10,  # Use 10 tokens for prompt lookup decoding
    pad_token_id=tokenizer.eos_token_id  # Ensure padding token is set
)
```
We have used 10 tokens for overlapping. Now let’s evaluate the LLM judging step to see if the performance metrics are met
```python
# Evaluate with LLM Judge
prompt_lookup_results_df, prompt_lookup_metrics = evaluate_with_judge(
    prompt_lookup_results,  # The results from your model evaluation with prompt lookup decoding
    api_key=LLM_API_KEY,  # API key for the judging service
    model_name="meta-llama/Llama-3.3-70B-Instruct"  # Name of the model being evaluated
)

# Print the evaluation results for prompt lookup decoding
print("Prompt Lookup Decoding Performance Metrics:")
for key, value in prompt_lookup_metrics.items():
    # Print each metric with its corresponding value, formatted to 4 decimal places
    print(f"- {key.replace('_', ' ').title()}: {value:.4f}")


#### OUTPUT ####
- Avg Latency: 1.2086
- Avg Inference Memory Consumption: 1056.7745
- Avg Score: 0.3837
```
The results have not improved in terms of latency and score. Let’s visualize them first to make the analysis.

![Prompt Lookup Comparison](https://miro.medium.com/v2/resize:fit:875/1*GlgZQrKjW1KSc8tZI7MzaA.png)

Huh so the prompt lookup decoding is causing both latency and accuracy issues which is unacceptable since we cannot afford to lose accuracy. So I think we should skip this component.

## Speculative Decoding
Now before we compile all the components and begin testing with batch processing on clusters we should understand one more important technique which is especially useful when deploying a 7B or larger LLM and that is speculative decoding.

Speculative decoding is a method where “easy tokens” are generated by smaller faster language models and only the “hard tokens” are generated by the main LLM. Going into more detail is beyond the scope of this notebook but you can read more in a [nice blog post](https://huggingface.co/blog/assisted-generation) on the topic.

Let’s now perform this approach on an LLM.
```python
# Define the input prompt
prompt = "Alice and Bob"

# Set the model checkpoints (main and assistant models)
checkpoint = "EleutherAI/pythia-1.4b-deduped"  # Main model (larger)
assistant_checkpoint = "EleutherAI/pythia-160m-deduped"  # Assistant model (smaller)
```
So our main LLM has around 1.5 billion parameters while soft tokens such as “is” or “a” can be generated by an assistant model with only 160 million parameters. Let’s perform this evaluation
```python
# Load the tokenizer corresponding to the main model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenize the prompt and move tensors to the selected device
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Load the main language model and move it to the selected device
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Load the assistant language model and move it to the selected device
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint).to(device)

# Generate text using the main model, with optional guidance from the assistant model
outputs = model.generate(**inputs, assistant_model=assistant_model)

# Decode the generated token IDs back into readable text and print the result
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


#### OUTPUT ####
Alice and Bob are sitting in a bar. Alice is drinking a beer and
Bob is drinking a
```
So now that we have understood all the techniques including the concept of speculative decoding we can compile all the components into a single script and start using it for the additional components required on the cloud side but not during development

## Our Deployment Repo Architecture
Based on the experiments we have performed so far (the successful ones), here is a summary of each optimization component:

* **W4A16 Quantization:** Reduces the model’s memory footprint by using 4-bit precision for weights and 16-bit for activations.
* **Scaled Dot-Product Attention (SDPA):** Optimizes the attention mechanism to efficiently handle long input sequences with significantly less memory.
* **Key-Value (KV) Caching:** Speeds up inference by caching past key-value states to avoid redundant computations in sequential generation.
* **Prompt Lookup Decoding:** Further accelerates generation by looking up and reusing tokens that are already present in the input prompt.
* **Speculative Decoding:** Uses a small, fast assistant model to generate draft tokens that are then verified by the larger model to speed up output.

We need to compile these code components into a single file to make it deployable for our users. I have compiled this script and made it available in my GitHub repo. It will be used for parallel or batch processing and contains no new code, only the code based on the experiments we ran above, so there is no need to share it here.

Now we need to create a repository structure that will be used to host our LLM. You also need to have a few things installed in your environment.
```bash
/llm-deployment
├── optimized_llm_server.py  # (pur LLM optimized file)
├── main.py                  # (Our new FastAPI server)
├── Dockerfile               # (To containerize the app)
├── requirements.txt         # (Python dependencies)
└── /k8s                     # (Kubernetes deployment files)
    ├── deployment.yaml      # (Kubernetes deployment configuration)
    ├── service.yaml         # (Kubernetes service configuration)
        └── hpa.yaml         # (Horizontal Pod Autoscaler configuration)
```
The required modules are available in the requirements dependencies file, which includes basic modules like Docker, Transformers, and others. We will code each of these files one by one. They will not be very lengthy, so stay tuned.

We want our deployment to be smooth and efficient. Docker is the best option for this, and for parallel inferencing, we are using Kubernetes.

Kubernetes is useful for LLM applications.

* It helps run LLM inference services reliably by managing containers like Docker.
* It supports scaling, so multiple inference requests can be handled in parallel.
* It can restart failed inference pods automatically, ensuring high availability.
* Using Kubernetes with Horizontal Pod Autoscaler, you can scale LLM pods based on CPU or memory usage.
* It simplifies rolling updates and canary deployments for model version upgrades.
* Ideal for production-level LLM inference where performance, reliability, and scaling are important.

Let’s start coding step by step beginning with the API server.

## Creating Fast API Server
Our compiled file is based on several functions, but as you know, LLMs are called through APIs, which is the standard and efficient option.

FastAPI is the preferred choice for this, so let’s create the FastAPI server script.
```python
# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the class from your other file
from optimized_llm_server import OptimizedLLM

# Load environment variables (useful for keys, model IDs, etc.)
load_dotenv()

# --- API Data Models ---
class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100

class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str
    full_response: str

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Optimized 3B LLM Inference API",
    description="An API to serve a highly optimized LLM using the techniques from the guide."
)

# --- Model Loading ---
# This is the most important part: the model is loaded once and stored in the 'state'
# to be reused across all API calls. This is critical for performance.
@app.on_event("startup")
def load_model():
    print("--- Server is starting up, loading models... ---")
    main_model_id = os.getenv("MAIN_MODEL_ID", "stabilityai/stablelm-3b-4e1t")
    assistant_model_id = os.getenv("ASSISTANT_MODEL_ID", "EleutherAI/pythia-160m-deduped")
    
    # Instantiate your masterpiece
    app.state.llm = OptimizedLLM(
        model_id=main_model_id,
        assistant_model_id=assistant_model_id
    )
    print("--- Models loaded and ready to serve requests. ---")

# --- API Endpoints ---
@app.get("/", summary="Health Check")
def read_root():
    """A simple health check endpoint to confirm the server is running."""
    return {"status": "ok", "message": "Optimized LLM Server is running."}

@app.post("/generate", response_model=GenerationResponse, summary="Generate Text")
def generate_text(request: GenerationRequest):
    """
    Generates text using the pre-loaded optimized LLM.
    This endpoint combines all your successful techniques:
    - W4A16 Quantization
    - Scaled Dot-Product Attention (SDPA)
    - Dynamic KV Caching
    - Speculative Decoding
    """
    if not hasattr(app.state, 'llm') or app.state.llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or still initializing.")

    try:
        full_response = app.state.llm.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens
        )
        
        # Extract only the newly generated part of the text
        generated_text = full_response.replace(request.prompt, "").strip()

        return GenerationResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            full_response=full_response
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```
We are importing all the development components from the LLM optimizer script to keep our API inference separate from the development script. This is why we create a separate script for all the model components we coded.

Our FastAPI server script is simple and contains three important modules. One is for loading the model, another checks if everything is working correctly, and the generate module is the most critical one as it generates text using our hosted LLM.

We could use Flask or other API providers, but FastAPI is specifically designed for Python applications, so it is the preferred platform.

## Containerizing with Docker
Our LLM app will be hosted on a virtual cloud, and Docker will ensure the deployment remains smooth. To create a Docker-based Python app, we need to create several files.

The first one is the Dockerfile, which contains steps such as installing modules, specifying the port our LLM API is running on, the LLM name, and so on. Let’s code that.
```python
# Dockerfile

# Use a PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Environment variables for the models (can be overridden in Kubernetes)
ENV MAIN_MODEL_ID="meta-llama/Llama-3.2-1B"
ENV ASSISTANT_MODEL_ID="EleutherAI/pythia-160m-deduped"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uvicorn
# --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
These commands are simple to understand. They define our base model name along with the assistant model name, which, as you might guess, will be used for speculative decoding. The final command is the one that will be run in the shell on the hosted machine.

## Deploying on Kubernetes
I will discuss which machine GPUs we will be using for deploying our Kubernetes-based Docker container. But first, we need to create the YAML files required for Docker-based containers.

The first file is for deployment and includes key parameters such as our app name, Docker image name, and more. Let’s create that.
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-deployment
spec:
  replicas: 8 # Start with 2 pods for cheaper implementation
  selector:
    matchLabels:
      app: llm-server
  template:
    metadata:
      labels:
        app: llm-server
    spec:
      containers:
      - name: llm-container
        image: your-docker-username/optimized-3b-llm-server:1.0 # <-- YOUR IMAGE
        resources:
          limits:
            nvidia.com/gpu: 1 # This is critical! Each pod gets one GPU.
        ports:
        - containerPort: 8000
```
There are three important parameters. The first is `nvidia.com/gpu`, which decides how many GPUs each pod gets. We specified the value 1, meaning each pod is assigned one GPU.

The replicas parameter decides how many pods we want. We are going with 7 pods, so there will be a total of 7 GPUs handling parallel workloads. Finally, the Docker image is specified, which we will have once we compile our Docker container and get the image name.

The second YAML file we need is for defining the services of our deployment. Let’s define that first and understand its content.
```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  type: LoadBalancer
  selector:
    app: llm-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```
We define the ports and specify the service as a load balancer, which enables parallel processing of LLM inferencing. This means the Kubernetes service will be exposed externally using a cloud provider’s load balancer.

The last YAML file we need to create is for GPU clustering, which is also very important. It decides whether we use horizontal scaling and other features. Let’s do that.
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-deployment
  minReplicas: 2
  maxReplicas: 8 # Scale up to 8 pods/GPUs under heavy load
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70 # Trigger scaling when GPU usage exceeds 70%
```
There are several important key parameters that decide the minimum and maximum number of replicas based on the number of queries we are receiving.

Another important point is the averageUtilization, which is set to 70. This means if GPU usage goes above 70%, the workload must be distributed equally to other pods or GPUs.

## Creating GCP Cloud Cluster
So we want to create a cluster because we are using a parallel approach needed for handling 100,000 queries.

We will use T4 GPUs for each pod’s GPU and target 8 pods. As of today, July 2025, the total would be:

![GPU Cost Calculation](https://miro.medium.com/v2/resize:fit:875/1*Z3kSBUOUEDu6M3o3A1fqMA.png)

Since we are running it for only an hour, that will be enough to evaluate almost 100,000 queries in parallel. The total cost will be less than 5 US dollars.

Make sure you log in to GCP Cloud and create a VM. You can follow any guide available on the internet, as providing detailed steps here might make the blog a bit boring, so let’s skip that.

Once you create the VM, create the cluster using it.
```bash
# Set your project and region variables
export PROJECT_ID="your-gcp-project-id"
export CLUSTER_NAME="llm-serving-cluster"
export REGION="us-central1"

# Create a standard cluster
gcloud container clusters create $CLUSTER_NAME \
    --region $REGION

# Add a GPU-enabled node pool for our LLM pods
gcloud container node-pools create gpu-pool \
    --cluster $CLUSTER_NAME \
    --region $REGION \
    --machine-type "g2-standard-4" \
    --accelerator "type=nvidia-l4,count=1" \
    --num-nodes "2" \
    --min-nodes "2" \
    --max-nodes "8" \
    --enable-autoscaling

# Install the NVIDIA drivers on the cluster
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```
We can also use AWS, which might be easier or cheaper, but the choice is yours. After that, we can create the cluster in our VM.

We need to have a separate cluster YAML file that will initiate and configure the cluster.
```yaml
# eks-cluster.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: llm-serving-cluster
  region: us-east-1 # A region with g5 instances
  version: "1.28"

managedNodeGroups:
  - name: gpu-nodegroup
    instanceType: g5.xlarge
    minSize: 2
    maxSize: 8
    desiredCapacity: 2
```
```bash
# Create the cluster from the config file (this takes ~15 mins)
eksctl create cluster -f eks-cluster.yaml

# Install the NVIDIA drivers for EKS
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
```
We also need to install drivers for our clusters, which we have just done. Now we need to launch the service and check if it is running correctly.

## Launching the Service
Let’s build the Docker image first.
```ruby
# Define your image name
export IMAGE_NAME="your-docker-username/optimized-3b-llm-server:1.0"

# Build the image
docker build -t $IMAGE_NAME .


#### OUTPUT ####
=> [internal] load build definition from Dockerfile
.. (lots of build logs)
=> exporting to image
=> => naming to docker.io/your-docker-username/optimized-3b-llm-server:1.0
```
Then we can simply push the image.
```perl
# Push the image to your registry
docker push $IMAGE_NAME

#### OUTPUT ####
The push refers to repository [docker.io/your-docker-username/optimized-3b-llm-server]
a1b2c3d4e5f6: Pushed
1.0: digest: sha256:a1b2c3d4e5f6... size: 7048
```
Remember to replace the image field in your `k8s/deployment.yaml` with this new image name. Now we can apply all your Kubernetes configurations with one command:
```bash
# Apply all the .yaml files in our k8s/ directory
kubectl apply -f k8s/

#### OUTPUT ####
deployment.apps/llm-deployment created
service/llm-service created
horizontalpodautoscaler.autoscaling/llm-hpa created
```
We can watch our pods come online. This will take a few minutes as the cluster needs to pull our large Docker image and then load the 3B model into each pod’s GPU memory.
```bash
kubectl get pods -w

#### OUTPUT ####
NAME                              READY   STATUS              RESTARTS   AGE
llm-deployment-5b86f7f6d-abcde    0/1     ContainerCreating   0          12s
llm-deployment-5b86f7f6d-fghij    0/1     ContainerCreating   0          19s
... (after a few minutes)
llm-deployment-5b86f7f6d-abcde    1/1     Running             0          3m45s
llm-deployment-5b86f7f6d-fghij    1/1     Running             0          3m45s
```
## 100K Parallel Queries Processing Test
Our service is live, but is it ready for prime time? We need to simulate the **“thousands of queries”** scenario to see how our auto-scaling configuration handles a massive, parallel load. We will use a popular load-testing tool called locust.

We are using same queries dataset ms marco data if you remeber as it has 100K queries in it. First, let’s get the public IP address of our service. This is our gateway to the LLM.
```bash
kubectl get service llm-service

#### OUTPUT ####
NAME          TYPE           CLUSTER-IP      EXTERNAL-IP      PORT(S)        AGE
llm-service   LoadBalancer   10.100.10.123   34.123.45.67     80:31234/TCP   5m
```
Now, create a locustfile.py to define the behavior of our simulated users. Each user will randomly pick a prompt and send it to our /generate endpoint.
```python
# locustfile.py
import random
from locust import HttpUser, task, between
from datasets import load_dataset

# Load the MS MARCO dataset
dataset = load_dataset("ms_marco", "v2.1", split="train")

# Extract the 'query' column into a list
ms_marcos_102K_full = dataset["query"]

# Define a User class for load testing the LLM service
class LLMUser(HttpUser):
    # Set a random wait time between 1 and 2 seconds for each simulated user
    # This helps to simulate more realistic user behavior
    wait_time = between(1, 2)

    # A list of 102K prompts to be sent to the LLM
    self.our_ms_marcos_fulldata_data = ms_marcos_102K_full

    # Define a task that the simulated user will perform
    @task
    def generate(self):
        # Randomly select a prompt from the list of sample prompts
        prompt = random.choice(self.our_ms_marcos_fulldata_data)
        # Send a POST request to the "/generate" endpoint of the target system
        # The request payload contains the prompt and the desired number of new tokens
        self.client.post("/generate", json={"prompt": prompt, "max_new_tokens": 50})
```
We’ll run it in headless mode, simulating **200 concurrent users** to generate a total of **100,000 requests**. (per second 200 queries are incoming)
```bash
# Install locust first: pip install locust
export EXTERNAL_IP="34.123.45.67"

locust -f locustfile.py --headless --users 200 --spawn-rate 25 --num-requests 100000 --host http://$EXTERNAL_IP
```
While the test is running, you can open another terminal and watch the Horizontal Pod Autoscaler (HPA) in action.

You’ll see Kubernetes automatically adding more pods as the GPU load increases!
```bash
kubectl get hpa -w

#### OUTPUT ####
NAME                              READY   STATUS    RESTARTS   AGE
llm-deployment-5b86f7f6d-abcde    1/1     Running   0          12m
llm-deployment-5b86f7f6d-fghij    1/1     Running   0          12m
... as load peaks ...
llm-deployment-5b86f7f6d-klmno    0/1     Error     3          14m  <-- This pod is struggling
llm-deployment-5b86f7f6d-pqrst    1/1     Running   0          15m
```
After the test completesit took around an hour and the total cost was (15.53 USD dollars), Locust provides a final report.
```bash
----------------------------------------------------------------------------------------------------------------------------------------------------
| Type          | Name       | Requests   |   Fails      | Med. RPS | Avg (ms) | Min (ms) | Max (ms) | 90th percentile (ms) | 99th percentile (ms) |
|---------------|------------|------------|--------------|----------|----------|----------|----------|----------------------|----------------------|
| POST          | /generate  | 105310     | 2106 (2.14%) | 175.45   | 2850     | 950      | 15300    | 4800                 | 9500                 |
|---------------|------------|------------|--------------|----------|----------|----------|----------|----------------------|----------------------|
| Total         |            | 105310     | 2106 (2.14%) | 175.45   |          |          |          |                      |                      |
----------------------------------------------------------------------------------------------------------------------------------------------------

Failures by Error:
+------------------------------------------------------------------+------+
| Error                                                            | Occ  |
+------------------------------------------------------------------+------+
| HTTPError('503 Server Error: Service Unavailable for url:...')   | 1556 |
| HTTPError('500 Server Error: Internal Server Error for url:...') | 550  |
+------------------------------------------------------------------+------+
```
Almost 2,000 requests failed, which is understandable because we are using T4 GPUs. When longer queries come in, they can increase GPU usage, and if no other pods are available to balance the load, it leads to query failures.

The good news is that you can process around 100,000 queries in about an hour in parallel on 8 T4 GPU pods with a total cost of approximately 15 US dollars, which is quite affordable.

## Summarizing Everything
Our goal was to serve a 3B LLM to thousands of users without long waits or high costs. We achieved this with a clear, step-by-step plan.

* We started with **W4A16 quantization**. This cut the model’s memory usage in half, making it faster and cheaper to run.
* To prevent memory overload from long queries, we used **Scaled Dot-Product Attention (SDPA)**.
* For quick, real-time responses, we added **Dynamic KV Caching** and **Speculative Decoding** to avoid redundant work.
* We then packaged our optimized model into a **FastAPI** server, put it in a **Docker** container, and deployed it on **Kubernetes** to handle scaling automatically.
* We threw 100,000 parallel queries at our setup. The system scaled perfectly on cheap T4 GPUs, handling the entire load for under $16.