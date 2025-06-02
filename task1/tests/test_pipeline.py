import unittest
import pandas as pd
import numpy as np
from src.preprocess import TextPreprocessor, DataPreprocessor
from src.features import FeatureEngineer, FeatureProcessor
from src.model import MultiTaskModel
from src.entity_extractor import EntityExtractor, EntityProcessor

class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor()
        self.sample_text = "Hello! This is a test message with some numbers 123 and special characters @#$."
    
    def test_clean_text(self):
        cleaned_text = self.preprocessor.clean_text(self.sample_text)
        self.assertIsInstance(cleaned_text, str)
        self.assertTrue(cleaned_text.islower())
        self.assertTrue(all(c.isalpha() or c.isspace() for c in cleaned_text))
    
    def test_tokenize(self):
        tokens = self.preprocessor.tokenize(self.sample_text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(token, str) for token in tokens))
    
    def test_remove_stopwords(self):
        tokens = self.preprocessor.tokenize(self.sample_text)
        filtered_tokens = self.preprocessor.remove_stopwords(tokens)
        self.assertIsInstance(filtered_tokens, list)
        self.assertLessEqual(len(filtered_tokens), len(tokens))
    
    def test_lemmatize(self):
        tokens = self.preprocessor.tokenize(self.sample_text)
        lemmatized_tokens = self.preprocessor.lemmatize(tokens)
        self.assertIsInstance(lemmatized_tokens, list)
        self.assertEqual(len(lemmatized_tokens), len(tokens))

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.engineer = FeatureEngineer()
        self.sample_texts = [
            "This is a test message.",
            "Another test message here.",
            "Testing the feature engineering."
        ]
    
    def test_extract_text_features(self):
        features = self.engineer.extract_text_features(self.sample_texts)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], len(self.sample_texts))
    
    def test_encode_labels(self):
        issue_types = ["bug", "feature", "bug"]
        urgency_levels = ["high", "low", "medium"]
        encoded_issues, encoded_urgency = self.engineer.encode_labels(issue_types, urgency_levels)
        self.assertIsInstance(encoded_issues, np.ndarray)
        self.assertIsInstance(encoded_urgency, np.ndarray)
        self.assertEqual(len(encoded_issues), len(issue_types))
        self.assertEqual(len(encoded_urgency), len(urgency_levels))

class TestEntityExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = EntityExtractor()
        self.sample_text = "John Smith works at Microsoft in Seattle. The project deadline is next Monday."
    
    def test_extract_entities(self):
        entities = self.extractor.extract_entities(self.sample_text)
        self.assertIsInstance(entities, dict)
        self.assertTrue(all(isinstance(entities[entity_type], list) 
                          for entity_type in entities))
    
    def test_process_batch(self):
        texts = [self.sample_text, "Another test message."]
        entities_list = self.extractor.process_batch(texts)
        self.assertIsInstance(entities_list, list)
        self.assertEqual(len(entities_list), len(texts))
        self.assertTrue(all(isinstance(entities, dict) for entities in entities_list))

class TestMultiTaskModel(unittest.TestCase):
    def setUp(self):
        self.model = MultiTaskModel(n_estimators=10)  # Use small number for testing
        self.X = np.random.rand(10, 5)  # Sample features
        self.y = np.column_stack((
            np.random.randint(0, 3, 10),  # Issue types
            np.random.randint(0, 2, 10)   # Urgency levels
        ))
    
    def test_train(self):
        metrics = self.model.train(self.X, self.y)
        self.assertIsInstance(metrics, dict)
        self.assertIn('issue_type', metrics)
        self.assertIn('urgency_level', metrics)
    
    def test_predict(self):
        self.model.train(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertIsInstance(predictions, tuple)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0].shape[0], self.X.shape[0])
        self.assertEqual(predictions[1].shape[0], self.X.shape[0])

def main():
    unittest.main()

if __name__ == '__main__':
    main() 