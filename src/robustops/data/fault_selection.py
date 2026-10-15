"""Parse JSON-only fault selector responses."""
import json


def parse_message(m):
    if m.get('reasoning_content'):
        raise ValueError('unexpected thinking')
    raw = m.get('content', '').strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
    return json.loads(raw)
