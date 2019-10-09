import sys

try:
    __import__(f"{sys.argv[1]}.main")
except Exception:
    print("Pls, read how to run scripts in `README.md`")
    raise
