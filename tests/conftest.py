import os

# Prevent JAX / XLA from preallocating all system RAM on CI runners
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("KERAS_BACKEND", "jax")
