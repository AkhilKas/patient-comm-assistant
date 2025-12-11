"""Medical text simplifier using open-source LLMs."""

import os
from typing import Optional
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .prompts import SYSTEM_PROMPT, SIMPLIFY_PROMPT, ANSWER_PROMPT


@dataclass
class SimplificationResult:
    """Container for simplification results."""
    original_text: str
    simplified_text: str
    model_used: str
    readability_before: Optional[dict] = None
    readability_after: Optional[dict] = None


class MedicalSimplifier:
    """
    Two-stage medical text simplifier using open-source LLMs.
    
    Recommended models (by VRAM/RAM requirement):
    - 8GB+:  "microsoft/Phi-3-mini-4k-instruct" (3.8B, fast)
    - 16GB+: "mistralai/Mistral-7B-Instruct-v0.3"
    - 16GB+: "meta-llama/Meta-Llama-3.1-8B-Instruct" (needs HF token)
    - 32GB+: "mistralai/Mixtral-8x7B-Instruct-v0.1"
    """
    
    MODEL_PRESETS = {
        "phi3": "microsoft/Phi-3-mini-4k-instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "llama3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "gemma": "google/gemma-2-2b-it",
    }
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: Optional[str] = None,
        use_preset: Optional[str] = None,
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the simplifier.
        
        Args:
            model_name: HuggingFace model name
            device: Device ('cpu', 'cuda', 'mps'). Auto-detects if None.
            use_preset: Use preset ('phi3', 'mistral', 'llama3', 'gemma')
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
            max_new_tokens: Max tokens to generate
            hf_token: HuggingFace token (needed for gated models like Llama)
        """
        if use_preset and use_preset in self.MODEL_PRESETS:
            model_name = self.MODEL_PRESETS[use_preset]
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        print(f"Loading model: {model_name}")
        print(f"Device: {device}")
        
        # Load tokenizer
        token = hf_token or os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings
        model_kwargs = {
            "token": token,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
        }
        
        if load_in_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model_kwargs["device_map"] = "auto"
        elif device != "cpu":
            model_kwargs["device_map"] = device
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Move to device if needed (for CPU or explicit device)
        if device == "cpu" or (device == "mps" and "device_map" not in model_kwargs):
            self.model = self.model.to(device)
        
        # Create pipeline for easier generation
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        print(f"✓ Model loaded successfully!")
    
    def _generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """Generate text from a prompt."""
        # Format for chat models
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        # Generate
        outputs = self.pipe(formatted, return_full_text=False)
        return outputs[0]["generated_text"].strip()
    
    def simplify_text(self, text: str) -> SimplificationResult:
        """
        Simplify medical text to 8th-grade reading level.
        
        Args:
            text: Medical text to simplify
            
        Returns:
            SimplificationResult with original and simplified text
        """
        prompt = SIMPLIFY_PROMPT.format(text=text)
        simplified = self._generate(prompt)
        
        return SimplificationResult(
            original_text=text,
            simplified_text=simplified,
            model_used=self.model_name
        )
    
    def answer_question(self, question: str, context: str) -> str:
        """
        Answer a patient question using retrieved context.
        
        Args:
            question: Patient's question
            context: Retrieved context from vector store
            
        Returns:
            Simplified answer
        """
        prompt = ANSWER_PROMPT.format(question=question, context=context)
        return self._generate(prompt)
    
    def simplify_with_verification(self, text: str) -> SimplificationResult:
        """
        Two-stage simplification with readability verification.
        
        Stage 1: Initial simplification
        Stage 2: Check and re-simplify if needed
        """
        # Stage 1: Initial simplification
        result = self.simplify_text(text)
        
        # Stage 2: Verification (optional re-simplification)
        verification_prompt = f"""Review this simplified medical text. Is it easy enough for someone with an 8th-grade education to understand? If it's too complex, make it simpler. If it's already simple enough, return it unchanged.

Text:
---
{result.simplified_text}
---

Final version:"""
        
        final_text = self._generate(verification_prompt)
        result.simplified_text = final_text
        
        return result


def create_simplifier(
    use_preset: str = "phi3",
    load_in_4bit: bool = False
) -> MedicalSimplifier:
    """
    Factory function to create a simplifier.
    
    Args:
        use_preset: Model preset ('phi3', 'mistral', 'llama3', 'gemma')
        load_in_4bit: Use 4-bit quantization (CUDA only)
    
    Examples:
        # Lightweight model for quick testing
        simplifier = create_simplifier(use_preset="phi3")
        
        # Better quality with Mistral
        simplifier = create_simplifier(use_preset="mistral")
    """
    return MedicalSimplifier(use_preset=use_preset, load_in_4bit=load_in_4bit)