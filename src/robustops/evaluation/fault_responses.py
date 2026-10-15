import json


def body(response):
    for block in response.get("result", {}).get("content", []):
        if block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except (ValueError, TypeError):
                pass
    return response.get("result", {}).get("structuredContent")


def collection(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                isinstance(child, list)
                and child
                and all(isinstance(x, dict) for x in child)
            ):
                return key, child
    return None, []


def text_body(r):
    for block in r.get("result", {}).get("content", []):
        if block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except (ValueError, TypeError):
                pass
    return None
