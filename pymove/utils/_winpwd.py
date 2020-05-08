class WindowsUser:
  def __init__(self):
    self.pw_name = 'windows'


def getpwuid(uid):
  return WindowsUser()
