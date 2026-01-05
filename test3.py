
import sqlite3

conn = sqlite3.connect("lightroom catalogue/lightroom catalogue.lrcat")
cursor = conn.cursor()

# # List all tables
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# tables = cursor.fetchall()
# for table in tables:
#     print(table[0])
cursor.execute("""
    SELECT text 
    FROM Adobe_imageDevelopSettings
    WHERE text IS NOT NULL
    LIMIT 200
""")

for row in cursor.fetchall():
    settings = row[0]
    print(settings)

conn.close()
