"""Lossless SQLite snapshot data transport through simple seed SQL parsers."""
import math
import re
import sqlite3


def literal(value):
    if value is None:
        return 'NULL'
    if isinstance(value, str):
        # Native seed endpoints split on lines/semicolons and SQLAlchemy text()
        # recognizes colon parameters even inside quoted values. Hex TEXT keeps
        # original bytes without exposing those delimiters to that parser.
        return "CAST(X'" + value.encode('utf-8').hex() + "' AS TEXT)"
    if isinstance(value, bytes):
        return "X'" + value.hex() + "'"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return 'NULL'
        if math.isinf(value):
            return '-9.0e999' if value < 0 else '9.0e999'
        return repr(value)
    raise TypeError('unsupported SQLite storage type')


def dump_database(connection):
    emitted = set()
    for statement in connection.iterdump():
        match = re.match(r'INSERT INTO "((?:""|[^"])*)"', statement)
        if not match:
            yield statement
            continue
        quoted_name = match.group(1)
        if quoted_name in emitted:
            continue
        emitted.add(quoted_name)
        for row in connection.execute('SELECT * FROM "' + quoted_name + '"'):
            yield 'INSERT INTO "' + quoted_name + '" VALUES(' + ','.join(literal(v) for v in row) + ');'


def transport_sql(snapshot):
    connection = sqlite3.connect(':memory:')
    try:
        connection.executescript(snapshot)
        objects = connection.execute("SELECT type,name FROM sqlite_master WHERE type IN ('view','table') AND name NOT LIKE 'sqlite_%' ORDER BY type DESC,name").fetchall()
        drops = ['DROP ' + kind.upper() + ' IF EXISTS "' + name.replace('"', '""') + '";' for kind, name in objects]
        return 'PRAGMA foreign_keys=OFF;\n' + '\n'.join(drops) + '\n' + '\n'.join(dump_database(connection))
    finally:
        connection.close()
