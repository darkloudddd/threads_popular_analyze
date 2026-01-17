import unittest
from src.login import LoginHandler

class TestLoginHandler(unittest.TestCase):

    def setUp(self):
        self.login_handler = LoginHandler()

    def test_successful_login(self):
        result = self.login_handler.perform_login('valid_username', 'valid_password')
        self.assertTrue(result)

    def test_failed_login(self):
        result = self.login_handler.perform_login('invalid_username', 'invalid_password')
        self.assertFalse(result)

    def test_empty_username(self):
        result = self.login_handler.perform_login('', 'some_password')
        self.assertFalse(result)

    def test_empty_password(self):
        result = self.login_handler.perform_login('some_username', '')
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()