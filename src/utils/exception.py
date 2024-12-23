class AppException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class InvalidInputException(AppException):
    def __init__(self, message="Invalid input provided"):
        super().__init__(message)
