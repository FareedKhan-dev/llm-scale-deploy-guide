import random
from locust import HttpUser, task, between

class LLMUser(HttpUser):
    wait_time = between(1, 3)

    sample_prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a short story about a robot who discovers music.",
        "What are the top 5 benefits of using Kubernetes?",
        "Translate 'Hello, how are you?' to French.",
        "Summarize the plot of the movie Inception.",
    ]

    @task
    def generate(self):
        prompt = random.choice(self.sample_prompts)
        payload = {
            "prompt": prompt,
            "max_new_tokens": 75
        }
        self.client.post("/generate", json=payload, name="/generate")