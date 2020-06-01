class WindowsUser:
    """Mock class for module errors on windows."""

    def __init__(self, uid):
        """Instantiates mock class."""
        self.pw_name = 'windows'
        self.uid = uid


def getpwuid(uid):
    """Default value for pwd on windows."""
    return WindowsUser(uid)
