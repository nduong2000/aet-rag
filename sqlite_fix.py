# sqlite_fix.py
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
