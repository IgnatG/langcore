"""Simple test for the custom provider plugin."""

import os

import dotenv

# Import the provider to trigger registration with LangCore
# Note: This manual import is only needed when running without installation.
# After `pip install -e .`, the entry point system handles this automatically.
from langcore_provider_example import CustomGeminiProvider  # noqa: F401

import langcore as lx


def main():
    """Test the custom provider."""
    dotenv.load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LANGCORE_API_KEY")

    if not api_key:
        print("Set GEMINI_API_KEY or LANGCORE_API_KEY to test")
        return

    config = lx.factory.ModelConfig(
        model_id="gemini-2.5-flash",
        provider="CustomGeminiProvider",
        provider_kwargs={"api_key": api_key},
    )
    model = lx.factory.create_model(config)

    print(f"✓ Created {model.__class__.__name__}")

    # Test inference
    prompts = ["Say hello"]
    results = list(model.infer(prompts))

    if results and results[0]:
        print(f"✓ Inference worked: {results[0][0].output[:50]}...")
    else:
        print("✗ No response")


if __name__ == "__main__":
    main()
