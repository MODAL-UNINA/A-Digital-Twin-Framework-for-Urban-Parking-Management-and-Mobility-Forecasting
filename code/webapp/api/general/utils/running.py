import sys


def is_main_running() -> bool:
    if "runserver" in sys.argv:
        return True

    if any("gunicorn" in v for v in sys.argv):
        return True

    return False
