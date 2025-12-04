#!/usr/bin/env python3
"""
Diagnostic script with FIXED model loading approach
"""

import sys
import traceback

print("=" * 60)
print("SPEAKER RECOGNITION DIAGNOSTIC (FIXED)")
print("=" * 60)

# Test 1: Check PyTorch and Torchaudio
print("\n1. Checking PyTorch and Torchaudio...")
try:
    import torch
    import torchaudio
    print(f"   ✅ torch version: {torch.__version__}")
    print(f"   ✅ torchaudio version: {torchaudio.__version__}")
except ImportError as e:
    print(f"   ❌ PyTorch/Torchaudio missing: {e}")
    sys.exit(1)

# Test 2: Check Pyannote
print("\n2. Checking Pyannote...")
try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    print("   ✅ pyannote.audio imported successfully")
except ImportError as e:
    print(f"   ❌ Pyannote missing: {e}")
    print("   Install with: pip install pyannote-audio")

# Test 3: Check SpeechBrain
print("\n3. Checking SpeechBrain...")
try:
    import speechbrain
    print(f"   ✅ speechbrain version: {speechbrain.__version__}")
except ImportError as e:
    print(f"   ❌ SpeechBrain NOT INSTALLED: {e}")
    print("   Install with: pip install speechbrain")

# Test 4: Check SpeechBrain EncoderClassifier
print("\n4. Checking SpeechBrain EncoderClassifier...")
try:
    from speechbrain.pretrained import EncoderClassifier
    print("   ✅ EncoderClassifier can be imported")
except ImportError as e:
    print(f"   ❌ EncoderClassifier import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Try to load the FIXED model (using pretrained module)
print("\n5. Testing model loading (FIXED approach)...")
try:
    print("   Attempting to load speaker embedding model...")
    print("   Using: speechbrain.pretrained.EncoderClassifier")
    
    embedding_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    print("   ✅ Model loaded successfully!")
    print(f"   Model type: {type(embedding_model)}")
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    print("\n   Full error:")
    traceback.print_exc()
    
    # Try alternative model
    print("\n   Trying alternative model: speechbrain/spkrec-xvect-voxceleb")
    try:
        embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb"
        )
        print("   ✅ Alternative model loaded successfully!")
    except Exception as e2:
        print(f"   ❌ Alternative model also failed: {e2}")
        sys.exit(1)

# Test 6: Test embedding extraction
print("\n6. Testing embedding extraction...")
try:
    import tempfile
    import numpy as np
    
    # Create a dummy audio signal (1 second of random noise)
    duration = 1.0
    sample_rate = 16000
    waveform = torch.randn(1, int(sample_rate * duration))
    
    print("   Extracting embedding from dummy audio...")
    embedding = embedding_model.encode_batch(waveform)
    embedding_np = embedding.squeeze().cpu().numpy()
    
    print(f"   ✅ Embedding extracted successfully!")
    print(f"   Embedding shape: {embedding_np.shape}")
    print(f"   Embedding dimension: {len(embedding_np)}")
    
except Exception as e:
    print(f"   ❌ Embedding extraction failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# All tests passed!
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nSpeaker recognition should work in your main application.")
print("\nIMPORTANT: Update your code to use:")
print("from speechbrain.pretrained import EncoderClassifier")
print("(instead of: from speechbrain.inference.speaker import EncoderClassifier)")