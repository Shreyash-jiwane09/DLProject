"""
Unit tests for utility functions
"""

import unittest
import os
import sys
from pathlib import Path
import json
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cnnClassifier.utils.common import (
    read_yaml,
    create_directories,
    save_json,
    load_json,
    get_size
)


class TestCommonUtils(unittest.TestCase):
    """Test cases for common utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_create_directories(self):
        """Test directory creation"""
        dirs = [
            self.test_dir / "dir1",
            self.test_dir / "dir2" / "subdir"
        ]
        
        create_directories(dirs, verbose=False)
        
        for dir_path in dirs:
            self.assertTrue(dir_path.exists())
            self.assertTrue(dir_path.is_dir())
    
    def test_save_and_load_json(self):
        """Test JSON save and load operations"""
        json_path = self.test_dir / "test.json"
        test_data = {
            "accuracy": 0.95,
            "loss": 0.12,
            "classes": ["class1", "class2"]
        }
        
        # Save JSON
        save_json(json_path, test_data)
        self.assertTrue(json_path.exists())
        
        # Load JSON
        loaded_data = load_json(json_path)
        self.assertEqual(loaded_data.accuracy, test_data["accuracy"])
        self.assertEqual(loaded_data.loss, test_data["loss"])
        self.assertEqual(loaded_data.classes, test_data["classes"])
    
    def test_get_size(self):
        """Test file size calculation"""
        test_file = self.test_dir / "test.txt"
        
        # Create a file with known size
        with open(test_file, 'w') as f:
            f.write('a' * 1024)  # 1 KB
        
        size_str = get_size(test_file)
        self.assertIsInstance(size_str, str)
        self.assertIn("KB", size_str)


class TestYAMLOperations(unittest.TestCase):
    """Test YAML operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.yaml_file = self.test_dir / "test.yaml"
    
    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_read_yaml(self):
        """Test YAML reading"""
        # Create test YAML file
        yaml_content = """
        model:
          name: VGG16
          classes: 2
        training:
          epochs: 25
          batch_size: 16
        """
        
        with open(self.yaml_file, 'w') as f:
            f.write(yaml_content)
        
        # Read YAML
        config = read_yaml(self.yaml_file)
        
        self.assertEqual(config.model.name, "VGG16")
        self.assertEqual(config.model.classes, 2)
        self.assertEqual(config.training.epochs, 25)
        self.assertEqual(config.training.batch_size, 16)


if __name__ == '__main__':
    unittest.main()
