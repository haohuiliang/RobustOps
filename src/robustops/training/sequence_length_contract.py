"""Distinguish verified lengths from explicitly deferred training-time lengths."""


def check_sequence_length(actual, expected, max_length, status='verified'):
    deferred = status == 'deferred_to_training' and expected == -1
    if not deferred and actual != expected:
        raise ValueError(f'sequence token drift: expected {expected}, found {actual}')
    if actual > max_length:
        raise ValueError(f'sequence_length={actual} exceeds max_length={max_length}')
