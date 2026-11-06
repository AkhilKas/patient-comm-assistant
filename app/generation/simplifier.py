"""Medical text simplifier using open-source LLMs."""

import importlib.util
import os
import platform
import sys
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseSimplifier, SimplificationResult  # noqa: F401 (re-exported)
from .errors import InputTooLongError
from .prompts import SYSTEM_PROMPT


def mlx_available() -> bool:
    """True on Apple Silicon with mlx-lm installed."""
    return (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    )


class MedicalSimplifier(BaseSimplifier):
    """
    Two-stage medical text simplifier using open-source LLMs.

    Two backends:
    - "torch": HuggingFace transformers on CUDA, MPS or CPU. 4-bit
      quantization (load_in_4bit) is available on CUDA only.
    - "mlx": Apple MLX via mlx-lm, using pre-quantized 4-bit models from
      mlx-community. Much faster and lighter on Apple Silicon.

    Recommended models (by VRAM/RAM requirement, torch fp16):
    - 8GB+:  "microsoft/Phi-3-mini-4k-instruct" (3.8B, fast)
    - 16GB+: "mistralai/Mistral-7B-Instruct-v0.2"
    - 16GB+: "meta-llama/Meta-Llama-3.1-8B-Instruct" (needs HF token)
    The 4-bit MLX versions need roughly a quarter of that.
    """

    MODEL_PRESETS = {
        "phi3": "microsoft/Phi-3-mini-4k-instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "llama3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "gemma": "google/gemma-2-2b-it",
    }

    MLX_MODEL_PRESETS = {
        "phi3": "mlx-community/Phi-3-mini-4k-instruct-4bit",
        "mistral": "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
        "llama3": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "gemma": "mlx-community/gemma-2-2b-it-4bit",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_preset: Optional[str] = None,
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
        hf_token: Optional[str] = None,
        backend: str = "auto",
    ):
        """
        Initialize the simplifier.

        Args:
            model_name: HuggingFace model name (an MLX-format repo when
                backend is "mlx"). Ignored if use_preset is given.
            device: Torch device ('cpu', 'cuda', 'mps'). Auto-detects if None.
            use_preset: Use preset ('phi3', 'mistral', 'llama3', 'gemma')
            load_in_4bit: Torch backend only; 4-bit quantization on CUDA
                (requires bitsandbytes). MLX presets are already 4-bit.
            max_new_tokens: Max tokens to generate
            hf_token: HuggingFace token (needed for gated models like Llama)
            backend: "auto" (MLX on Apple Silicon if installed, else torch),
                "mlx" or "torch"
        """
        if backend == "auto":
            backend = "mlx" if mlx_available() else "torch"
        if backend not in ("mlx", "torch"):
            raise ValueError(f"Unknown backend {backend!r}; use 'auto', 'mlx' or 'torch'")
        if backend == "mlx" and not mlx_available():
            raise RuntimeError("MLX backend needs Apple Silicon and `pip install mlx-lm`")

        presets = self.MLX_MODEL_PRESETS if backend == "mlx" else self.MODEL_PRESETS
        if use_preset:
            if use_preset not in presets:
                raise ValueError(f"Unknown preset {use_preset!r}; choose from {sorted(presets)}")
            model_name = presets[use_preset]
        model_name = model_name or presets["phi3"]

        self.backend = backend
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        token = hf_token or os.getenv("HF_TOKEN")

        print(f"Loading model: {model_name}")
        if backend == "mlx":
            context_length = self._load_mlx(model_name, token)
        else:
            context_length = self._load_torch(model_name, device, load_in_4bit, token)

        # Input budget: model context minus room for the generated tokens
        self.max_input_tokens = context_length - max_new_tokens

        print(f"✓ Model loaded successfully!")

    def _load_torch(self, model_name: str, device: Optional[str], load_in_4bit: bool,
                    token: Optional[str]) -> int:
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        print(f"Backend: torch ({device})")

        # All preset models are natively supported by transformers, so remote
        # code is not needed. (Phi-3's remote code is incompatible with the
        # KV cache in recent transformers versions.)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate settings
        model_kwargs = {
            "token": token,
            "dtype": torch.float16 if device != "cpu" else torch.float32,
        }

        if load_in_4bit:
            if device == "cuda":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                model_kwargs["device_map"] = "auto"
            else:
                print("⚠ load_in_4bit is CUDA-only with torch; loading in "
                      "full precision (use the MLX backend for 4-bit on Mac)")

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Quantized models are already placed by device_map
        if "device_map" not in model_kwargs:
            self.model = self.model.to(device)

        self.model.eval()
        return getattr(self.model.config, "max_position_embeddings", 4096)

    def _load_mlx(self, model_name: str, token: Optional[str]) -> int:
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.device = "mlx"
        print("Backend: mlx")

        # mlx-lm downloads via huggingface_hub, which reads HF_TOKEN from the env
        if token:
            os.environ.setdefault("HF_TOKEN", token)

        self.model, self.tokenizer, config = load(model_name, return_config=True)
        self._mlx_sampler = make_sampler(temp=0.7, top_p=0.9)
        return config.get("max_position_embeddings", 4096)

    def _format_prompt(self, prompt: str, system_prompt: str) -> tuple[str, bool]:
        """
        Format the prompt with the model's chat template.

        Returns the formatted string and whether it already contains the
        special tokens (so they are not added a second time when tokenizing).
        """
        if not getattr(self.tokenizer, "chat_template", None):
            return f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:", False

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Some templates (Gemma, Mistral v0.2) reject the system role;
            # fold the system prompt into the user turn instead.
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{prompt}"}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return formatted, True

    def _generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """Generate text from a prompt."""
        formatted, has_special_tokens = self._format_prompt(prompt, system_prompt)

        # No truncation: cutting the prompt would drop the generation marker
        # at the end and produce garbage, so reject oversized inputs instead.
        input_ids = self.tokenizer.encode(formatted, add_special_tokens=not has_special_tokens)
        if len(input_ids) > self.max_input_tokens:
            raise InputTooLongError(
                f"Input is {len(input_ids)} tokens; this model accepts at most "
                f"{self.max_input_tokens}. Please shorten the text."
            )

        if self.backend == "mlx":
            from mlx_lm import generate
            response = generate(
                self.model,
                self.tokenizer,
                prompt=input_ids,
                max_tokens=self.max_new_tokens,
                sampler=self._mlx_sampler,
            )
            return response.strip()

        inputs = torch.tensor([input_ids], device=self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        generated_tokens = outputs[0][len(input_ids):]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()


def create_simplifier(
    use_preset: str = "phi3",
    load_in_4bit: bool = False,
    backend: str = "auto",
) -> MedicalSimplifier:
    """
    Factory function to create a simplifier.

    Args:
        use_preset: Model preset ('phi3', 'mistral', 'llama3', 'gemma')
        load_in_4bit: Use 4-bit quantization (torch backend, CUDA only)
        backend: 'auto', 'mlx' or 'torch'

    Examples:
        # Lightweight model for quick testing
        simplifier = create_simplifier(use_preset="phi3")

        # Better quality with Mistral
        simplifier = create_simplifier(use_preset="mistral")

        # Force PyTorch on a Mac (e.g. to compare against MLX)
        simplifier = create_simplifier(backend="torch")
    """
    return MedicalSimplifier(use_preset=use_preset, load_in_4bit=load_in_4bit, backend=backend)
