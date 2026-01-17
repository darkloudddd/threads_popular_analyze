import unittest
from src.parser import Parser

class TestParser(unittest.TestCase):

    def setUp(self):
        self.parser = Parser()

    def test_parse_words(self):
        test_data = ["apple", "banana", "apple", "orange", "banana", "banana"]
        expected_result = {"apple": 2, "banana": 3, "orange": 1}
        result = self.parser.parse_words(test_data)
        self.assertEqual(result, expected_result)

    def test_parse_empty_list(self):
        test_data = []
        expected_result = {}
        result = self.parser.parse_words(test_data)
        self.assertEqual(result, expected_result)

    def test_parse_single_word(self):
        test_data = ["apple"]
        expected_result = {"apple": 1}
        result = self.parser.parse_words(test_data)
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()