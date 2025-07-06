import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

class OptimizedLLM:
    """
    A class to load and run an LLM with W4A16 Quantization, SDPA, 
    Dynamic KV Caching, and Speculative Decoding.
    """
    def __init__(self, model_id: str, assistant_model_id: str = None, device: str = "cuda"):
        self.model_id = model_id
        self.assistant_model_id = assistant_model_id
        self.device = device
        self.model = None
        self.assistant_model = None
        self.tokenizer = None
        self._load_model()

    def _get_quantization_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    def _load_model(self):
        print(f"Loading tokenizer for {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = self._get_quantization_config()

        print(f"Loading main model: {self.model_id} with W4A16, SDPA...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        if self.assistant_model_id:
            print(f"Loading assistant model for speculative decoding: {self.assistant_model_id}...")
            self.assistant_model = AutoModelForCausalLM.from_pretrained(
                self.assistant_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

        print("Models loaded successfully.")

    def generate(self, prompt: str, max_new_tokens: int = 100):
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded first.")

        model_inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        generation_kwargs = {
            **model_inputs,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        if self.assistant_model:
            generation_kwargs["assistant_model"] = self.assistant_model

        print("Generating response...")
        start_time = time.time()

        output = self.model.generate(**generation_kwargs)

        latency = time.time() - start_time
        print(f"Inference latency: {latency:.4f} seconds")

        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return decoded_output