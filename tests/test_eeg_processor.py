"""
Unit tests for EEG processing functionality.

This module contains tests for the EEG processor, signal processing,
and feature extraction components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from backend.eeg_processing.processor import EEGProcessor, CircularBuffer


class TestCircularBuffer:
    """Test cases for CircularBuffer class."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = CircularBuffer(size=100, channels=4)
        assert buffer.size == 100
        assert buffer.channels == 4
        assert buffer.index == 0
        assert not buffer.is_full

    def test_add_sample(self):
        """Test adding samples to buffer."""
        buffer = CircularBuffer(size=10, channels=2)

        # Add first sample
        sample = np.array([1.0, 2.0])
        buffer.add_sample(sample)

        assert buffer.index == 1
        assert not buffer.is_full

        # Fill buffer
        for i in range(9):
            buffer.add_sample(sample)

        assert buffer.is_full
        assert buffer.index == 0

    def test_get_data_not_full(self):
        """Test getting data when buffer is not full."""
        buffer = CircularBuffer(size=10, channels=2)

        # Add 3 samples
        for i in range(3):
            buffer.add_sample(np.array([float(i), float(i + 1)]))

        data = buffer.get_data()
        assert data.shape == (2, 10)  # Full buffer size
        assert np.array_equal(data[:, :3], [[0, 1, 2], [1, 2, 3]])

    def test_get_latest_samples(self):
        """Test getting latest samples."""
        buffer = CircularBuffer(size=10, channels=2)

        # Fill buffer completely
        for i in range(12):  # Overflow to test circular behavior
            buffer.add_sample(np.array([float(i), float(i + 1)]))

        latest = buffer.get_latest_samples(5)
        expected = np.array([[7, 8, 9, 10, 11], [8, 9, 10, 11, 12]])
        assert np.array_equal(latest, expected)


class TestEEGProcessor:
    """Test cases for EEGProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create EEG processor instance."""
        return EEGProcessor(sampling_rate=256, channels=4, buffer_size=512)

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.fs == 256
        assert processor.channels == 4
        assert processor.buffer.size == 512
        assert processor.quality_threshold == 0.7

    def test_preprocess_signal(self, processor):
        """Test signal preprocessing."""
        # Create test signal
        t = np.linspace(0, 2, 512)  # 2 seconds
        signal_data = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        test_signal = np.tile(signal_data, (4, 1))  # 4 channels

        # Add some noise
        test_signal += np.random.normal(0, 0.1, test_signal.shape)

        processed = processor._preprocess_signal(test_signal)

        # Check that signal is filtered (should reduce high frequency noise)
        assert processed.shape == test_signal.shape
        assert processed.dtype == np.float64

    def test_calculate_band_powers(self, processor):
        """Test frequency band power calculation."""
        # Create test signal with known frequency components
        fs = 256
        t = np.linspace(0, 2, 512)
        test_signal = np.zeros((4, 512))

        # Add alpha wave (10 Hz)
        test_signal += 0.5 * np.sin(2 * np.pi * 10 * t)
        # Add beta wave (20 Hz)
        test_signal += 0.3 * np.sin(2 * np.pi * 20 * t)

        test_signal = np.tile(test_signal, (4, 1))

        band_powers = processor._calculate_band_powers(test_signal)

        # Check that we have power measurements for each band
        expected_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        for band in expected_bands:
            assert f"{band}_power_avg" in band_powers
            assert band_powers[f"{band}_power_avg"] >= 0

        # Alpha power should be relatively high due to our test signal
        assert band_powers['alpha_power_avg'] > band_powers['delta_power_avg']

    def test_calculate_attention_score(self, processor):
        """Test attention score calculation."""
        # Test with high beta, low alpha (high attention)
        band_powers = {
            'beta_power_avg': 1.0,
            'alpha_power_avg': 0.2,
        }
        attention = processor._calculate_attention_score(band_powers)
        assert attention > 0.5  # Should be high attention

        # Test with low beta, high alpha (low attention)
        band_powers = {
            'beta_power_avg': 0.2,
            'alpha_power_avg': 1.0,
        }
        attention = processor._calculate_attention_score(band_powers)
        assert attention < 0.5  # Should be low attention

    def test_calculate_stress_level(self, processor):
        """Test stress level calculation."""
        # Test with high beta/gamma ratio (high stress)
        band_powers = {
            'beta_power_avg': 1.0,
            'gamma_power_avg': 0.2,
            'theta_power_avg': 0.1,
        }
        stress = processor._calculate_stress_level(band_powers)
        assert stress > 0.5  # Should indicate stress

        # Test with low beta/gamma ratio (low stress)
        band_powers = {
            'beta_power_avg': 0.2,
            'gamma_power_avg': 1.0,
            'theta_power_avg': 0.5,
        }
        stress = processor._calculate_stress_level(band_powers)
        assert stress < 0.5  # Should indicate relaxation

    def test_assess_signal_quality(self, processor):
        """Test signal quality assessment."""
        # Create good quality signal
        t = np.linspace(0, 2, 512)
        good_signal = np.sin(2 * np.pi * 10 * t)
        good_signal = np.tile(good_signal, (4, 1))

        quality = processor._assess_signal_quality(good_signal)

        assert 'overall_quality' in quality
        assert 'avg_snr' in quality
        assert quality['overall_quality'] >= 0
        assert quality['overall_quality'] <= 1

    @pytest.mark.asyncio
    async def test_process_sample_empty_buffer(self, processor):
        """Test processing sample with insufficient buffer data."""
        sample = np.array([1.0, 2.0, 3.0, 4.0])

        result = await processor.process_sample(sample)

        # Should return empty response when buffer is not full enough
        assert result['attention_score'] == 0.5
        assert result['stress_level'] == 0.5
        assert result['signal_quality'] == 0.0


# Integration test
@pytest.mark.asyncio
async def test_eeg_processor_integration():
    """Integration test for EEG processor."""
    processor = EEGProcessor(sampling_rate=256, channels=4, buffer_size=1024)

    # Simulate real-time data processing
    sample_count = 600  # About 2.3 seconds at 256Hz

    for i in range(sample_count):
        # Generate realistic EEG-like signal
        t = i / 256.0
        # Mix of different frequency components
        sample = np.array([
            50e-6 * np.sin(2 * np.pi * 10 * t) + 20e-6 * np.sin(2 * np.pi * 40 * t),  # Alpha + Gamma
            40e-6 * np.sin(2 * np.pi * 12 * t) + 15e-6 * np.sin(2 * np.pi * 25 * t),  # Beta
            30e-6 * np.sin(2 * np.pi * 8 * t) + 25e-6 * np.sin(2 * np.pi * 15 * t),   # Alpha + Beta
            35e-6 * np.sin(2 * np.pi * 6 * t) + 10e-6 * np.sin(2 * np.pi * 35 * t),   # Theta + Gamma
        ])

        # Add some noise
        sample += np.random.normal(0, 5e-6, 4)

        result = await processor.process_sample(sample)

        # Should get valid results after buffer fills
        if i >= 128:  # After buffer has enough data
            assert 'attention_score' in result
            assert 'stress_level' in result
            assert 'cognitive_load' in result
            assert 'processing_time' in result
            assert result['processing_time'] > 0

    # Check final performance stats
    stats = processor.get_performance_stats()
    assert stats['processing_count'] > 0
    assert stats['avg_processing_time'] > 0
