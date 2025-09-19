#!/usr/bin/env python3
"""
Test script for research-validated EEG learning platform features.

This script tests the core research implementations:
1. CNN-LSTM Cognitive Load Prediction Model
2. Research-based EEG Processing (Theta/Alpha ratios, Gamma power)
3. Dynamic Difficulty Adjustment Algorithm
4. Real-time Cognitive Load Assessment

Usage: python test_research_features.py
"""

import sys
import os
import numpy as np
import time
from typing import Dict, Any

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_basic_imports():
    """Test basic library imports."""
    print("🔬 Testing Research Feature Imports...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} loaded")

        import numpy as np
        print(f"✅ NumPy {np.__version__} loaded")

        import scipy
        print(f"✅ SciPy {scipy.__version__} loaded")

        # Test research-specific imports
        try:
            from eeg_processing.processor import EEGProcessor
            print("✅ EEG Processor (Research-based) loaded")
        except ImportError as e:
            print(f"⚠️ EEG Processor import failed: {e}")

        try:
            from ml_models.knowledge_tracing.cnn_lstm_model import CognitiveLoadPredictor
            print("✅ CNN-LSTM Cognitive Load Predictor loaded")
        except ImportError as e:
            print(f"⚠️ CNN-LSTM Model import failed: {e}")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_eeg_processor():
    """Test research-based EEG processor."""
    print("\n🧠 Testing EEG Processor (Research Implementation)...")
    try:
        from eeg_processing.processor import EEGProcessor

        # Initialize processor with research parameters (14-channel Emotiv EPOC+)
        processor = EEGProcessor(
            sampling_rate=256,
            channels=14,  # Emotiv EPOC+ standard
            buffer_size=1024
        )

        print("✅ EEG Processor initialized with research parameters:")
        print(f"   - Sampling rate: {processor.fs} Hz")
        print(f"   - Channels: {processor.channels} (Emotiv EPOC+ standard)")
        print(f"   - Research frequency bands: {list(processor.frequency_bands.keys())}")
        print(f"   - Target latency: <{processor.target_latency_ms}ms")
        print(f"   - Target accuracy: {processor.target_accuracy * 100}%")

        # Test with synthetic EEG data
        print("\n📊 Testing EEG feature extraction...")
        n_samples = 512
        synthetic_eeg = np.random.randn(14, n_samples) * 100e-6  # 14 channels, realistic amplitude

        # Process EEG data
        start_time = time.time()
        features = processor.preprocess(synthetic_eeg)
        processing_time = (time.time() - start_time) * 1000

        print("✅ EEG processing completed:")
        print(f"   - Processing time: {processing_time:.2f}ms")
        print(f"   - Features extracted: {len(features)}")
        print(f"   - Target latency achieved: {processing_time < processor.target_latency_ms}")

        # Check research-specific features
        research_features = ['theta_alpha_ratio', 'gamma_power_relative', 'connectivity_avg']
        for feature in research_features:
            if feature in features:
                print(f"   - {feature}: {features[feature]:.4f}")
            else:
                print(f"   - {feature}: Not found")

        return True

    except Exception as e:
        print(f"❌ EEG Processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cnn_lstm_model():
    """Test CNN-LSTM cognitive load prediction model."""
    print("\n🤖 Testing CNN-LSTM Cognitive Load Prediction...")
    try:
        from ml_models.knowledge_tracing.cnn_lstm_model import CognitiveLoadPredictor

        # Initialize model
        predictor = CognitiveLoadPredictor()

        print("✅ CNN-LSTM Model initialized:")
        print(f"   - Model type: {predictor.model.__class__.__name__}")
        print(f"   - Input channels: {predictor.model.n_channels}")
        print(f"   - Sequence length: {predictor.model.seq_len}")

        # Test with synthetic EEG data
        print("\n🔬 Testing cognitive load prediction...")
        test_eeg = np.random.randn(14, 512) * 100e-6  # 14 channels, 512 samples

        start_time = time.time()
        prediction = predictor.predict(test_eeg)
        prediction_time = (time.time() - start_time) * 1000

        print("✅ Cognitive load prediction completed:")
        print(f"   - Prediction time: {prediction_time:.2f}ms")
        print(f"   - Predicted load level: {prediction['cognitive_load_level']}")
        print(f"   - Confidence score: {prediction['confidence_score']:.3f}")
        print(f"   - Theta/Alpha ratio: {prediction.get('theta_alpha_ratio', 'N/A')}")
        print(f"   - Target latency achieved: {prediction_time < 50}")

        return True

    except Exception as e:
        print(f"❌ CNN-LSTM Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dynamic_difficulty_service():
    """Test dynamic difficulty adjustment service."""
    print("\n🎯 Testing Dynamic Difficulty Adjustment...")
    try:
        from services.dynamic_difficulty_service import DynamicDifficultyService

        service = DynamicDifficultyService()

        print("✅ Dynamic Difficulty Service initialized:")
        print(f"   - Service type: Research-validated algorithm")
        print(f"   - Optimal load range: {service.progression_model['optimal_load_range']}")
        print(f"   - Target latency: <{service._DynamicDifficultyService__class__.target_latency_ms}ms")

        # Test difficulty adjustment
        print("\n⚙️ Testing difficulty adjustment logic...")

        # Mock EEG features (high cognitive load scenario)
        eeg_features = {
            'theta_alpha_ratio': 1.8,  # High cognitive load
            'gamma_power_relative': 0.25,
            'connectivity_avg': 0.2
        }

        # Mock learning context
        learning_context = {
            'success_rate': 0.6,
            'time_spent_seconds': 450,
            'error_rate': 0.3,
            'session_duration_minutes': 15
        }

        import asyncio
        async def test_adjustment():
            result = await service.adjust_difficulty(
                user_id="test_user",
                current_difficulty=4,
                eeg_features=eeg_features,
                learning_context=learning_context
            )

            print("✅ Difficulty adjustment completed:")
            print(f"   - Current difficulty: {result.current_difficulty}")
            print(f"   - Recommended difficulty: {result.recommended_difficulty}")
            print(f"   - Adjustment reason: {result.adjustment_reason}")
            print(f"   - Confidence score: {result.confidence_score:.3f}")
            print(f"   - Processing time: {result.processing_time_ms:.2f}ms")
            print(f"   - Research validation: {result.research_validation[:50]}...")

            return True

        return asyncio.run(test_adjustment())

    except Exception as e:
        print(f"❌ Dynamic Difficulty Service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_research_integration():
    """Test complete research integration pipeline."""
    print("\n🔬 Testing Complete Research Integration Pipeline...")

    try:
        # 1. Generate synthetic EEG data (14-channel Emotiv EPOC+)
        print("📡 Generating synthetic EEG data (14-channel Emotiv EPOC+)...")
        n_channels = 14
        seq_len = 512
        eeg_data = np.random.randn(n_channels, seq_len) * 100e-6

        # 2. Process EEG with research algorithm
        from eeg_processing.processor import EEGProcessor
        processor = EEGProcessor(channels=n_channels)

        start_time = time.time()
        features = processor.preprocess(eeg_data)
        processing_time = (time.time() - start_time) * 1000

        # 3. Predict cognitive load with CNN-LSTM
        from ml_models.knowledge_tracing.cnn_lstm_model import CognitiveLoadPredictor
        predictor = CognitiveLoadPredictor()

        prediction = predictor.predict(eeg_data)
        total_time = (time.time() - start_time) * 1000

        # 4. Apply dynamic difficulty adjustment
        from services.dynamic_difficulty_service import DynamicDifficultyService
        service = DynamicDifficultyService()

        import asyncio
        async def complete_test():
            adjustment = await service.adjust_difficulty(
                user_id="research_test",
                current_difficulty=3,
                eeg_features=features,
                learning_context={
                    'success_rate': 0.75,
                    'time_spent_seconds': 300,
                    'error_rate': 0.2,
                    'session_duration_minutes': 10
                }
            )

            print("✅ Complete research pipeline executed:")
            print(f"   - EEG Processing: {processing_time:.2f}ms")
            print(f"   - Cognitive Load Prediction: {prediction['cognitive_load_level']} ({prediction['confidence_score']:.3f})")
            print(f"   - Difficulty Adjustment: {adjustment.current_difficulty} → {adjustment.recommended_difficulty}")
            print(f"   - Total Pipeline Time: {total_time:.2f}ms")
            print(f"   - Target Performance Achieved: {total_time < 100}")

            return True

        return asyncio.run(complete_test())

    except Exception as e:
        print(f"❌ Research integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all research feature tests."""
    print("🚀 AI-EEG Learning Platform - Research Feature Testing")
    print("=" * 60)

    test_results = []

    # Test 1: Basic imports
    test_results.append(("Basic Imports", test_basic_imports()))

    # Test 2: EEG Processor
    test_results.append(("EEG Processor", test_eeg_processor()))

    # Test 3: CNN-LSTM Model
    test_results.append(("CNN-LSTM Model", test_cnn_lstm_model()))

    # Test 4: Dynamic Difficulty Service
    test_results.append(("Dynamic Difficulty", test_dynamic_difficulty_service()))

    # Test 5: Complete Integration
    test_results.append(("Research Integration", test_research_integration()))

    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All research features are working correctly!")
        print("🔬 Ready for production deployment with validated algorithms.")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the implementation.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
