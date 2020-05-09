class WindowsUser:
  def __init__(self):
    self.pw_name = "windows"
    self.ru_maxrss = 0

RUSAGE_SELF = 0

def getpwuid(uid):
  return WindowsUser()

def getrusage(RUSAGE_SELF):
  return WindowsUser()
