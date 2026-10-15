"""Keep every validation record under veRL's fixed-batch loss reduction."""
import math


class FullValidationDataset:
    """Zero-pad batches and correct their denominator without duplicating loss.

    veRL averages fixed-global-batch losses across validation batches. Scaling
    real row masks by padded_count / real_count makes that reduction the exact
    mean over real records, including the final incomplete batch.
    """

    def __init__(self, dataset, global_batch_size, expected_records=None):
        self.dataset = dataset
        self.real_count = getattr(dataset, '_real_length', len(dataset))
        if self.real_count <= 0:
            raise ValueError('Validation requires at least one record')
        if expected_records is not None and self.real_count != expected_records:
            raise ValueError(f'Validation expects {expected_records} records, got {self.real_count}')
        if global_batch_size <= 0:
            raise ValueError('Global batch size must be positive')
        self.padded_count = math.ceil(self.real_count / global_batch_size) * global_batch_size

    def __len__(self):
        return self.padded_count

    def __getitem__(self, index):
        if not 0 <= index < self.padded_count:
            raise IndexError(index)
        item = dict(self.dataset[min(index, self.real_count - 1)])
        scale = self.padded_count / self.real_count if index < self.real_count else 0.0
        item['loss_mask'] = item['loss_mask'] * scale
        return item
