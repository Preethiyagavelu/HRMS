import sqlite3

def init_db():
    conn = sqlite3.connect('database/interns.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS interns (
            intern_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()
