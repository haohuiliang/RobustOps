"""Compare restored SQLite state without treating DDL layout as data drift."""
import sqlite3


def state_contents(sql):
    connection = sqlite3.connect(':memory:')
    try:
        connection.executescript(sql)
        result = []
        objects = connection.execute("SELECT type,name,tbl_name FROM sqlite_master ORDER BY type,name").fetchall()
        for kind, name, owner in objects:
            quoted = '"' + name.replace('"', '""') + '"'
            if kind == 'table':
                result.append((kind, name,
                    connection.execute('PRAGMA table_info(' + quoted + ')').fetchall(),
                    connection.execute('PRAGMA foreign_key_list(' + quoted + ')').fetchall(),
                    connection.execute('SELECT * FROM ' + quoted).fetchall()))
            elif kind == 'index':
                result.append((kind, name, owner,
                    connection.execute('PRAGMA index_xinfo(' + quoted + ')').fetchall(),
                    connection.execute('SELECT sql FROM sqlite_master WHERE name=?', (name,)).fetchone()))
            else:
                result.append((kind, name, owner,
                    connection.execute('SELECT sql FROM sqlite_master WHERE name=?', (name,)).fetchone()))
        return result
    finally:
        connection.close()
