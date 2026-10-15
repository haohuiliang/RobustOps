"""Lossless references and mechanical checks for terminal evidence.

This module proves properties of *evidence*, not the truth of natural-language
claims. Matching a sentence to a field, interpreting a tool's effects, and
deciding whether a resource identity is complete remain semantic obligations.
"""
from copy import deepcopy
from dataclasses import dataclass
import json


class Number(str):
    """A JSON number's original token, without float rounding."""


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("non-JSON number: " + value)


def decode(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_int=Number,
                      parse_float=Number, parse_constant=_constant)


def atom(value):
    """JSON-safe, type-sensitive value; 1, true and '1' are distinct."""
    if isinstance(value, Number):
        return {"type": "number", "value": str(value)}
    if value is None:
        return {"type": "null", "value": None}
    if type(value) is bool:
        return {"type": "boolean", "value": value}
    if type(value) in (int, float):
        return {"type": "number", "value": json.dumps(value, allow_nan=False)}
    if type(value) is str:
        return {"type": "string", "value": value}
    if type(value) in (dict, list):
        if value:
            raise ValueError("an atomic reference cannot be a nonempty container")
        return {"type": "object" if isinstance(value, dict) else "array", "value": value}
    raise TypeError("unsupported source value")


def validate_atom(value):
    if not isinstance(value, dict) or set(value) != {"type", "value"}:
        raise ValueError("expected must be a typed literal")
    kind, literal = value["type"], value["value"]
    if kind == "number":
        if type(literal) is not str or not isinstance(decode(literal), Number):
            raise ValueError("number literal requires a JSON number token")
    elif kind == "string" and type(literal) is str:
        pass
    elif kind == "boolean" and type(literal) is bool:
        pass
    elif kind == "null" and literal is None:
        pass
    elif kind == "object" and type(literal) is dict and not literal:
        pass
    elif kind == "array" and type(literal) is list and not literal:
        pass
    else:
        raise ValueError("invalid typed literal")


@dataclass
class EvidenceIndex:
    packet: dict
    nodes: dict
    events: dict
    view: dict

    def resolve(self, ref):
        """Resolve a generated reference, including explicit JSON-text hops."""
        value = self.packet
        for step in self.nodes[ref]["location"]:
            value = decode(value) if step is None else value[step]
        return value

    def equality(self, ref, expected):
        return atom(self.resolve(ref)) == expected

    def same_row(self, refs):
        """Require exact same immediate object, not merely a shared event."""
        if not refs:
            return False
        nodes = [self.nodes[ref] for ref in refs]
        return all(not node["whole_document"] and type(node["field"]) is str and
                   node["parent"] == nodes[0]["parent"] for node in nodes)

    def evaluate(self, check):
        """Execute a small evidence proposition; never promote a claim to true.

        Malformed checks are errors. Well-formed checks produce pass/fail/unknown
        with program-resolved evidence. Neither the expected value nor the
        chosen identity is trusted as a correct interpretation of language.
        """
        op = check.get("op")
        keys = {
            "equals": {"op", "ref", "expected"},
            "field_equals": {"op", "ref", "field", "expected"},
            "same_row": {"op", "refs"},
            "response_success": {"op", "event_id"},
            "latest_same_tool_observation": {"op", "ref", "identity_refs"},
        }
        if op not in keys or set(check) != keys[op]:
            raise ValueError("invalid evidence check contract")
        if op in ("equals", "field_equals"):
            actual = atom(self.resolve(check["ref"]))
            expected = check["expected"]
            validate_atom(expected)
            # Matching includes the JSON type. This is intentionally lexical
            # for numbers; numerical unit conversion requires semantic review.
            actual_field = next(step for step in reversed(self.nodes[check["ref"]]["location"])
                                if isinstance(step, str))
            field_matches = op == "equals" or actual_field == check["field"]
            return {"status": "pass" if actual == expected and field_matches else "fail",
                    "actual": actual, "actual_field": actual_field, "refs": [check["ref"]]}
        if op == "same_row":
            if (not isinstance(check["refs"], list) or len(check["refs"]) < 2 or
                    len(set(check["refs"])) != len(check["refs"])):
                raise ValueError("row binding requires at least two references")
            return {"status": "pass" if self.same_row(check["refs"]) else "fail",
                    "refs": check["refs"]}
        if op == "response_success":
            if type(check["event_id"]) is not int:
                raise ValueError("event ID must be an integer")
            event = self.events[check["event_id"]]
            return {"status": {"reported_success": "pass", "reported_error": "fail",
                               "unknown": "unknown"}[event["response_status"]],
                    "response_status": event["response_status"],
                    "refs": [event["observation"]["ref"]]}
        history = self.field_history(check["ref"], check["identity_refs"])
        successful = [item for item in history if item["response_status"] == "reported_success"]
        if not successful:
            return {"status": "unknown", "history": history, "refs": []}
        latest = successful[-1]
        # Conflicting rows in one response and later undecodable observations
        # must not silently select a convenient value.
        if any(item["event_id"] >= latest["event_id"] and
               (item["response_status"] == "unknown" or
                (item["response_status"] == "reported_success" and item["value"] != latest["value"]))
               for item in history):
            status = "unknown"
        else:
            status = "pass" if self.nodes[check["ref"]]["event_id"] == latest["event_id"] else "fail"
        return {"status": status, "history": history, "refs": [latest["ref"]],
                "scope": "observations of explicit identity through the same tool only"}

    def field_history(self, ref, identity_refs):
        """Find same-field observations of a caller-specified exact row identity.

        It deliberately does not equate aliases, infer an entity from an ID's
        suffix, join different resource types, or treat arguments as state.
        Only observations from the same tool are compared. The caller must
        establish that these identity fields identify the intended resource.
        Returned history is observed history, not a proof
        that no other operation can affect that resource.
        """
        node = self.nodes[ref]
        if node["role"] != "observation" or not identity_refs:
            raise ValueError("history requires an observation and explicit identity")
        if not self.same_row([ref, *identity_refs]):
            raise ValueError("identity and value must belong to the same object")
        if ref in identity_refs:
            raise ValueError("the changing field cannot identify its own history")
        keys = [self.nodes[r]["field"] for r in identity_refs]
        if any(type(k) is not str for k in keys) or len(keys) != len(set(keys)):
            raise ValueError("identity must use distinct object fields")
        identity = {k: atom(self.resolve(r)) for k, r in zip(keys, identity_refs)}
        # Exact field names provide a structural namespace, not an inferred
        # semantic resource type. Callers cannot use this as a global identity.
        candidates = []
        tool = self.events[node["event_id"]]["tool"]
        for candidate_ref, candidate in self.nodes.items():
            if (candidate["role"] != "observation" or candidate["whole_document"] or
                    candidate["field"] != node["field"] or
                    self.events[candidate["event_id"]]["tool"] != tool):
                continue
            siblings = {n["field"]: r for r, n in self.nodes.items()
                        if n["parent"] == candidate["parent"] and not n["whole_document"]}
            if all(k in siblings and atom(self.resolve(siblings[k])) == v for k, v in identity.items()):
                event = self.events[candidate["event_id"]]
                candidates.append({"ref": candidate_ref, "event_id": event["id"],
                                   "response_status": event["response_status"],
                                   "value": atom(self.resolve(candidate_ref))})
        return sorted(candidates, key=lambda item: item["event_id"])


def build_index(packet):
    """Build a complete evidence view with generated references, no model calls.

    The packet remains unchanged. Only MCP text blocks that actually contain
    JSON objects/arrays are decoded in the view. Their original bytes remain
    addressable. Ambiguous or invalid JSON stays opaque with a recorded issue.
    All other fields, including errors and empty results, remain visible.
    """
    packet = deepcopy(packet)
    nodes, events = {}, {}
    view = {"reference_format": "Each atomic evidence value is [ref, typed_value]. "
            "References identify source fields, not verified language claims.",
            "context": [], "events": []}

    def register(location, event_id, role, whole=False):
        ref = "f" + str(len(nodes) + 1)
        nodes[ref] = {"location": list(location), "parent": list(location[:-1]),
                      "field": location[-1], "event_id": event_id, "role": role,
                      "whole_document": whole}
        return ref

    def walk(value, location, event_id, role):
        if isinstance(value, dict) and value:
            return {k: walk(v, [*location, k], event_id, role) for k, v in value.items()}
        if isinstance(value, list) and value:
            return [walk(v, [*location, i], event_id, role) for i, v in enumerate(value)]
        ref = register(location, event_id, role)
        return [ref, atom(value)]

    for key in ("user_request", "runtime_policy", "used_tool_schemas", "calendar_facts"):
        ref = register([key], None, "context", whole=True)
        view["context"].append({"ref": ref, "kind": key, "value": packet[key]})

    for i, event in enumerate(packet["evidence"]):
        event_id = event["id"]
        if type(event_id) is not int or event_id in events:
            raise ValueError("evidence IDs must be unique integers")
        if events and event_id <= max(events):
            raise ValueError("evidence IDs must preserve execution order")
        loc = ["evidence", i]
        entry = {"id": event_id, "tool": event["tool"], "decode_issues": []}
        flags = []
        observation = event["observation"]
        for owner, field in ((observation, "is_error"),
                             (observation.get("result", {}), "isError")):
            if isinstance(owner, dict) and type(owner.get(field)) is bool:
                flags.append(owner[field])
        # Preserve payload failures separately from transport status.
        payload_failures = []
        if isinstance(observation.get("result"), dict) and observation["result"].get("success") is False:
            payload_failures.append(register([*loc, "observation", "result", "success"],
                                             event_id, "observation"))

        def observation_walk(value, location):
            if isinstance(value, dict):
                output = {}
                for key, child in value.items():
                    child_loc = [*location, key]
                    # Only MCP content text blocks have this decoding contract.
                    is_text = key == "text" and value.get("type") == "text"
                    if is_text and isinstance(child, str) and child.lstrip().startswith(("{", "[")):
                        raw_ref = register(child_loc, event_id, "observation", whole=True)
                        try:
                            parsed = decode(child)
                        except ValueError as exc:
                            entry["decode_issues"].append({"ref": raw_ref, "reason": str(exc)})
                            output[key] = [raw_ref, atom(child)]
                        else:
                            if isinstance(parsed, dict) and parsed.get("success") is False:
                                payload_failures.append(raw_ref)
                            output[key] = {"raw_ref": raw_ref,
                                           "decoded_json": walk(parsed, [*child_loc, None], event_id, "observation")}
                    else:
                        output[key] = observation_walk(child, child_loc)
                return output if value else walk(value, location, event_id, "observation")
            if isinstance(value, list) and value:
                return [observation_walk(v, [*location, j]) for j, v in enumerate(value)]
            return walk(value, location, event_id, "observation")

        args_ref = register([*loc, "arguments"], event_id, "arguments", whole=True)
        obs_ref = register([*loc, "observation"], event_id, "observation", whole=True)
        entry["arguments"] = {"ref": args_ref, "value": walk(event["arguments"], [*loc, "arguments"], event_id, "arguments")}
        entry["observation"] = {"ref": obs_ref, "value": observation_walk(observation, [*loc, "observation"])}
        entry["response_status"] = ("reported_error" if any(flags) or payload_failures else
                                    "unknown" if not flags or entry["decode_issues"] else "reported_success")
        entry["payload_failure_refs"] = payload_failures
        events[event_id] = entry
        view["events"].append(entry)
    return EvidenceIndex(packet=packet, nodes=nodes, events=events, view=view)
