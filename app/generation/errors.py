"""Generation errors (kept free of torch/transformers imports)."""


class InputTooLongError(ValueError):
    """Raised when a prompt does not fit in the model's context window."""


class LLMUnavailableError(RuntimeError):
    """The provider couldn't answer right now (timeout, rate limit, outage,
    blocked response). Safe to retry with a fallback provider."""


class LLMRequestError(RuntimeError):
    """The provider rejected the request (bad key, unknown model, invalid
    request). Retrying elsewhere would hide a configuration problem."""
