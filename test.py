import sqlite3

conn = sqlite3.connect("path/to/your/catalog.lrcat")
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for table in tables:
    print(table[0])

conn.close()