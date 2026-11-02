"""Generation errors (kept free of torch/transformers imports)."""


class InputTooLongError(ValueError):
    """Raised when a prompt does not fit in the model's context window."""
