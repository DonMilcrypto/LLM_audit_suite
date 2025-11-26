import os
import torch
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    # Model Configuration
    # Defaults to a small model for testing if env var not set
    MODEL_PATH: str = os.getenv("MODEL_PATH", "gpt2")
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Execution Optimization
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "8"))

    # Generation Hyperparameters
    # Using frozen dictionary for immutability
    GEN_PARAMS: dict = field(default_factory=lambda: {
        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "150")),
        "do_sample": True,
        "temperature": float(os.getenv("TEMPERATURE", "1.2")),
        "top_k": int(os.getenv("TOP_K", "50")),
        "top_p": float(os.getenv("TOP_P", "0.95")),
        "repetition_penalty": 1.2,
    })

    ITERATIONS_PER_PROMPT: int = int(os.getenv("ITERATIONS", "5"))
