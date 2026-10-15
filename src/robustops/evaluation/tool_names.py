"""Conventional read verbs with optional namespace prefixes, not a capability catalog.

These predicates establish a candidate name boundary only. Fault qualification
still requires exposed schemas, successful native results and mechanism evidence.
"""
import re

def is_read_name(name: str) -> bool:
    return bool(re.search(r'(?:^|_)(?:get|list|find|search|retrieve|check|fetch|query|count)_', name))

def is_getter_name(name: str) -> bool:
    return bool(re.search(r'(?:^|_)(?:get|retrieve)_', name))
