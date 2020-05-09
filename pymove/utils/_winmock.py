class WindowsUser:
  """Mock class for module errors on windows."""
  def __init__(self):
    """Instantiates mock class."""
    self.pw_name = "windows"
    self.ru_maxrss = 0

RUSAGE_SELF = 0

def getpwuid(uid):
  """Default value for pwd on windows."""
  return WindowsUser()

def getrusage(RUSAGE_SELF):
  """Default value for resources on windows."""
  return WindowsUser()
