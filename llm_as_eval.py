# Make sure to install the openai library if you haven't already
# !pip install openai

import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re

def setup_llm_judge(api_key=None):
    """
    Set up the LLM-based judge for scoring.
    
    Args:
        api_key (str, optional): API key for Nebius. If None, will try to get from environment.
    
    Returns:
        OpenAI client or None if API key not available
    """
    # Try to get API key from parameter or environment variable
    if api_key is None:
        api_key = os.environ.get("NEBIUS_API_KEY")
    
    # Return None if no API key is available
    if not api_key:
        return None
    
    # Initialize OpenAI client with Nebius API endpoint
    client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=api_key
    )
    return client

def extract_score_with_regex(response_text):
    """
    Extract score from LLM response using regex patterns.
    
    Args:
        response_text (str): Raw response from LLM
    
    Returns:
        float: Extracted score or 0.0 if no valid score found
    """
    # Try multiple regex patterns to extract score
    patterns = [
        r'^(\d+\.?\d*)$',  # Just a number (e.g., "0.8")
        r'(\d+\.?\d*)/1\.0',  # Fraction format (e.g., "0.8/1.0")
        r'Score:\s*(\d+\.?\d*)',  # "Score: 0.8"
        r'(\d+\.?\d*)\s*/\s*1\.0',  # With spaces (e.g., "0.8 / 1.0")
        r'(\d+\.?\d*)\s*$',  # Number at end of string
        r'(\d+\.?\d*)',  # Any number in the string (fallback)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text.strip())
        if match:
            try:
                score = float(match.group(1))
                # Ensure score is within valid range
                if 0.0 <= score <= 1.0:
                    return score
                # If score is out of range, try to normalize it
                elif score > 1.0 and score <= 10.0:
                    return score / 10.0
                elif score > 10.0 and score <= 100.0:
                    return score / 100.0
            except ValueError:
                continue
    
    return 0.0

def get_similarity_score(client, generated_answer, ground_truth_answer, model_name):
    """
    Uses an LLM to score the semantic similarity between two answers.
    
    Args:
        client: OpenAI client instance
        generated_answer (str): The generated answer to evaluate
        ground_truth_answer (str): The ground truth answer to compare against
        model_name (str): The name of the model to use for evaluation.
    
    Returns:
        tuple: (score, raw_response) - score between 0.0 and 1.0 and raw LLM response
    """

    # Define a highly lenient system prompt for the LLM judge
    SYSTEM_PROMPT = """
    You are a very generous and lenient AI evaluator. Your primary goal is to give the [Generated Answer] the benefit of the doubt. Focus on whether it makes any reasonable attempt to address the question, rather than on its accuracy or completeness.

    **Your Task:**
    1.  Read the [Generated Answer] and the [Ground Truth Answer].
    2.  Evaluate if the [Generated Answer] is a good-faith effort to provide a relevant response. Be very flexible.
    3.  Assign a score from 0.0 to 1.0 based on the highly lenient rubric below.
    4.  Your response MUST BE ONLY the floating-point number. Do not add any explanations.

    **Highly Lenient Scoring Rubric:**

    - **1.0 (Excellent Match):** The [Generated Answer] correctly addresses the core question, even if the phrasing is different, it's less detailed, or contains minor inaccuracies. If the intent is correct, score it high.

    - **0.8-0.9 (Good Match):** The [Generated Answer] is clearly relevant and makes a solid attempt to answer the question. It captures the main idea, even if it contains some factual errors or omits key details. Give credit for being on the right track.

    - **0.6-0.7 (Partial Match):** The [Generated Answer] is on-topic and shows some understanding of the question, but might be incomplete, partially incorrect, or less precise. It's a reasonable attempt that should be rewarded.

    - **0.4-0.5 (Related Concept):** The [Generated Answer] touches upon the topic of the question but doesn't directly answer it. It shows some relevance but misses the main point.

    - **0.2-0.3 (Barely Relevant):** The [Generated Answer] mentions keywords from the question but the response is largely off-topic or nonsensical.

    - **0.0-0.1 (Irrelevant):** The [Generated Answer] is completely unrelated to the question, provides no meaningful content, or is a refusal to answer.
    """
    # Format user message with the answers to be compared
    user_message = f"""
    Please evaluate the following pair of answers based on their semantic similarity.

    [Generated Answer]:
    {generated_answer}

    [Ground Truth Answer]:
    {ground_truth_answer}
    """
    
    try:
        # Make API call to LLM judge with specific parameters
        response = client.chat.completions.create(
            model=model_name,  # Use specified model for evaluation
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,  # Deterministic output for consistent scoring
            max_tokens=10     # Slightly increased tokens for better extraction
        )
        
        # Extract raw response for debugging
        raw_response = response.choices[0].message.content.strip()
        
        # Use regex to extract score
        score = extract_score_with_regex(raw_response)
        
        return score, raw_response
    except Exception as e:
        # Return 0.0 and error message if there's any error in scoring
        return 0.0, f"Error: {str(e)}"

def score_single_result(result, api_key, model_name):
    """
    Score a single result - helper function for parallel processing.
    
    Args:
        result (dict): Single result dictionary
        api_key (str): API key for the client
        model_name (str): The name of the model to use for evaluation.
    
    Returns:
        tuple: (score, raw_response) for maintaining order and debugging
    """
    client = setup_llm_judge(api_key)
    if not client:
        return 0.0, "No API key available"
    
    # Clean the generated answer by removing the question part for fair comparison
    clean_generated_answer = result['generated_answer'].replace(result['question'], '').strip()
    
    # Get similarity score and raw response from LLM judge
    score, raw_response = get_similarity_score(client, clean_generated_answer, result['ground_truth_answer'], model_name)
    return score, raw_response

def evaluate_with_judge(model_results, api_key=None, model_name="deepseek-ai/DeepSeek-V3", show_progress=True, max_workers=5):
    """
    Score the model results using LLM judge and calculate metrics with parallel processing.
    
    Args:
        model_results (list): Results from evaluate_model function
        api_key (str, optional): API key for Nebius. If None, will try to get from environment.
        model_name (str): The name of the model to use for evaluation.
        show_progress (bool): Whether to show progress bar
        max_workers (int): Maximum number of parallel workers for scoring
    
    Returns:
        tuple: (results_df, metrics_dict) or (results_df, None) if no API key
    """
    # Check if API key is available
    if api_key is None:
        api_key = os.environ.get("NEBIUS_API_KEY")
    
    if not api_key:
        results_df = pd.DataFrame(model_results)
        return results_df, None
    
    # Initialize scores and responses lists with None values to maintain order
    scores = [None] * len(model_results)
    raw_responses = [None] * len(model_results)
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(score_single_result, result, api_key, model_name): i 
            for i, result in enumerate(model_results)
        }
        
        # Set up progress bar if requested
        if show_progress:
            progress_bar = tqdm(total=len(model_results), desc="Scoring results")
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                score, raw_response = future.result()
                scores[index] = score
                raw_responses[index] = raw_response
                model_results[index]['similarity_score'] = score
                model_results[index]['llm_judge_response'] = raw_response
                
                if show_progress:
                    progress_bar.update(1)
            except Exception as e:
                scores[index] = 0.0
                raw_responses[index] = f"Exception: {str(e)}"
                model_results[index]['similarity_score'] = 0.0
                model_results[index]['llm_judge_response'] = f"Exception: {str(e)}"
                
                if show_progress:
                    progress_bar.update(1)
        
        if show_progress:
            progress_bar.close()
    
    # Convert results list to pandas DataFrame for easier analysis
    results_df = pd.DataFrame(model_results)
    
    # Calculate aggregate metrics across all results
    metrics = {
        'avg_latency': results_df['generation_time_s'].mean(),      # Average response time
        'avg_inference_memory_consumption': results_df['peak_memory_mb'].mean(),          # Average memory consumption
        'avg_score': results_df['similarity_score'].mean()         # Average similarity score
    }
    
    return results_df, metrics