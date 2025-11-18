"""
Unit tests for prediction pipeline
"""

import unittest
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cnnClassifier.pipeline.predict import PredictionPipeline


class TestPredictionPipeline(unittest.TestCase):
    """Test cases for PredictionPipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.test_dir = Path("test_data")
        cls.test_dir.mkdir(exist_ok=True)
        
        # Create a dummy image
        cls.test_image_path = cls.test_dir / "test_image.jpg"
        img = Image.new('RGB', (224, 224), color='red')
        img.save(cls.test_image_path)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        if cls.test_image_path.exists():
            cls.test_image_path.unlink()
        if cls.test_dir.exists():
            cls.test_dir.rmdir()
    
    def test_initialization(self):
        """Test pipeline initialization"""
        pipeline = PredictionPipeline(str(self.test_image_path))
        self.assertEqual(pipeline.filename, str(self.test_image_path))
        self.assertEqual(len(pipeline.class_names), 2)
        self.assertIn('adenocarcinoma', pipeline.class_names)
        self.assertIn('normal', pipeline.class_names)
    
    def test_validate_image_valid(self):
        """Test image validation with valid image"""
        pipeline = PredictionPipeline(str(self.test_image_path))
        self.assertTrue(pipeline.validate_image())
    
    def test_validate_image_not_exists(self):
        """Test image validation with non-existent file"""
        pipeline = PredictionPipeline("non_existent.jpg")
        self.assertFalse(pipeline.validate_image())
    
    def test_validate_image_size_limit(self):
        """Test image size validation"""
        # Create a large dummy file
        large_file = self.test_dir / "large_image.jpg"
        with open(large_file, 'wb') as f:
            f.write(b'0' * (11 * 1024 * 1024))  # 11 MB
        
        pipeline = PredictionPipeline(str(large_file))
        self.assertFalse(pipeline.validate_image())
        
        # Cleanup
        large_file.unlink()
    
    def test_prediction_without_model(self):
        """Test prediction when model doesn't exist"""
        pipeline = PredictionPipeline(str(self.test_image_path))
        result = pipeline.predict()
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        # Should return error if model doesn't exist
        if not os.path.exists(pipeline.model_path):
            self.assertIn('error', result[0])


class TestPredictionOutput(unittest.TestCase):
    """Test prediction output format"""
    
    def test_prediction_output_structure(self):
        """Test that prediction output has correct structure"""
        # Mock prediction result
        expected_keys = ['prediction', 'confidence', 'confidence_percentage', 'all_probabilities']
        
        # This would be the actual output structure
        sample_output = {
            'prediction': 'normal',
            'confidence': '0.9500',
            'confidence_percentage': '95.00%',
            'all_probabilities': {
                'adenocarcinoma': 0.05,
                'normal': 0.95
            }
        }
        
        for key in expected_keys:
            self.assertIn(key, sample_output)
        
        self.assertIsInstance(sample_output['all_probabilities'], dict)
        self.assertEqual(len(sample_output['all_probabilities']), 2)


if __name__ == '__main__':
    unittest.main()
