class LoginHandler:
    def __init__(self, session, login_url):
        self.session = session
        self.login_url = login_url

    def perform_login(self, username, password):
        payload = {
            'username': username,
            'password': password
        }
        response = self.session.post(self.login_url, data=payload)

        if response.ok and self.is_logged_in(response):
            print("Login successful!")
            return True
        else:
            print("Login failed!")
            return False

    def is_logged_in(self, response):
        # Check for a specific element in the response that indicates a successful login
        return "Welcome" in response.text  # Adjust this condition based on the actual response content