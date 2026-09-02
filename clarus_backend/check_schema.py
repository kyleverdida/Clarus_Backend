import sqlite3

conn = sqlite3.connect('clarus.db')
for row in conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='encounters'"):
    print(row[0])
conn.close()