"""
Basic test to check if imports work and models can be loaded.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules import successfully."""
    print("Testing imports...")

    try:
        from streaming_app.workers.generation_worker import GenerationWorker
        print("✓ GenerationWorker imported")
    except Exception as e:
        print(f"✗ Failed to import GenerationWorker: {e}")
        return False

    try:
        from streaming_app.models.streaming_inference import StreamingInferenceEngine
        print("✓ StreamingInferenceEngine imported")
    except Exception as e:
        print(f"✗ Failed to import StreamingInferenceEngine: {e}")
        return False

    try:
        from streaming_app.utils.visualization import FrameRenderer
        print("✓ FrameRenderer imported")
    except Exception as e:
        print(f"✗ Failed to import FrameRenderer: {e}")
        return False

    try:
        from streaming_app.websocket.session_manager import SessionManager
        print("✓ SessionManager imported")
    except Exception as e:
        print(f"✗ Failed to import SessionManager: {e}")
        return False

    print("\n✓ All imports successful!")
    return True


def test_config():
    """Test that config can be loaded."""
    print("\nTesting config loading...")

    try:
        import yaml
        config_path = Path(__file__).parent / "config" / "streaming_config.yaml"

        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"✓ Config loaded successfully")
        print(f"  - Audio sample rate: {config['streaming']['audio_sample_rate']}")
        print(f"  - Denoising steps: {config['streaming']['denoising_steps']}")
        print(f"  - Target FPS: {config['streaming']['target_fps']}")

        return True

    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False


def test_checkpoint_exists():
    """Test that the checkpoint file exists."""
    print("\nTesting checkpoint file...")

    checkpoint_path = Path("checkpoints/last.ckpt")
    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"✓ Checkpoint found: {checkpoint_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("  Please download checkpoints using:")
        print("  git clone https://huggingface.co/robinwitch/DyStream")
        return False


def test_visualization_tools():
    """Test that visualization tools exist."""
    print("\nTesting visualization tools...")

    vis_dir = Path("tools/visualization_0416")
    if not vis_dir.exists():
        print(f"✗ Visualization directory not found: {vis_dir}")
        return False

    config_path = vis_dir / "configs" / "head_animator_best_0506.yaml"
    if not config_path.exists():
        print(f"✗ Visualization config not found: {config_path}")
        return False

    print(f"✓ Visualization tools found")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("DyStream Streaming App - Basic Tests")
    print("=" * 60)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test config
    results.append(("Config", test_config()))

    # Test checkpoint
    results.append(("Checkpoint", test_checkpoint_exists()))

    # Test visualization tools
    results.append(("Visualization Tools", test_visualization_tools()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! Server should be ready to start.")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues before starting the server.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
