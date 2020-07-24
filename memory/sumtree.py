import operator
import numpy as np


class SegmentTree:
    """Segment tree using a linear numpy array. Works well with batch updates.
    Can be subclassed to create other trees based on operations for example a sumtree.
    """
    def __init__(self, size, operation):
        assert size % 2 == 0, f"Only works with even size."
        self.neutral_value = 0
        if operation is min:
            self.neutral_value = np.inf
        elif operation is max:
            self.neutral_value = -np.inf

        self.tree = np.array([self.neutral_value for _ in range(2*size - 1)], dtype=np.float)
        self.tree_size = len(self.tree)
        self.size = size
        self.entries = 0
        self.op = operation  # Allows for Sum or Min tree.
        self.idxs = None

    def __len__(self):
        return self.size

    def build(self, arr):
        for i in range(self.size):
            self.tree[self.size - 1 + i] = arr[i]

        start_idx = self.size
        for i in reversed(range(start_idx)[1:]):
            self.tree[i - 1] = self.op(self.tree[(i << 1) - 1], self.tree[(i << 1 | 1) - 1])
        self.entries = len(arr)

    @property
    def is_full(self) -> bool:
        return self.size == self.entries

    def _update(self, idx, priority) -> None:
        """Update priority at given index
        :param idx: index to replace priority
        :param priority: priority value
        :return: None
        """
        i = idx + self.size
        self.tree[i - 1] = priority
        while i > 1:
            self.tree[(i >> 1) - 1] = self.op(self.tree[i - 1], self.tree[(i ^ 1) - 1])
            i >>= 1

    def _batch_update(self, vals: np.ndarray) -> None:
        """Update memory in a vectorised manner.
        :param vals: new priorities
        :return: None
        """
        assert len(self.idxs) == len(vals)
        idxs = self.idxs
        self.idxs = None
        self.tree[idxs] = vals
        idxs += 1

        while len(idxs) > 0:
            self.tree[(idxs >> 1) - 1] = self.op(self.tree[idxs - 1], self.tree[(idxs ^ 1) - 1])
            idxs >>= 1
            idxs = idxs[idxs > 1]
            idxs = np.unique(idxs)


class SumTree(SegmentTree):
    def __init__(self, size):
        super(SumTree, self).__init__(size=size, operation=operator.add)
        self.sample_taken = False
        self.max_priority = 1.0
        self.min_priority = np.inf

    def _set_minmax(self, max_prior, min_prior):
        self.max_priority = max(self.max_priority, max_prior)
        self.min_priority = min(self.min_priority, min_prior)

    def build(self, arr):
        self._set_minmax(max(arr), min(arr))
        super().build(arr)

    @property
    def total_sum(self):
        return self.tree[0]

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s):
        idx = self._retrieve(0, s)
        return idx, self.tree[idx]

    def sample_low(self, eps=0.0001) -> int:
        """Samples an index based on reverse priority.
        :param eps: small value to ensure a valid prob
        :return int: index of priority to replace
        """
        probs = self.max_priority - self.tree[-self.size:] + eps
        probs /= probs.sum()
        idx = np.random.choice(a=int(self.size), p=probs)
        return idx

    def prioritized_sample(self, n):
        """Sample n samples from memory with prioritisation"""
        assert not self.sample_taken
        self.sample_taken = True
        self.idxs = []
        priorities = []
        zero_idx = self.size - 1

        for _ in range(n):
            s = np.random.uniform(0.0, self.total_sum)
            idx, p = self.get(s)
            priorities.append(p)
            self._update(idx - zero_idx, 0.0)
            self.idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.total_sum
        self.idxs = np.array(self.idxs)
        return self.idxs - zero_idx, sampling_probabilities

    def update_weights(self, weights):
        """Update weights by batch. Reduces complexity from O(k log N) to O(log N) since we work with vectors."""
        self._set_minmax(weights.max(), weights.min())
        self.sample_taken = False
        self._batch_update(weights)

    def append(self, idx, priority=None) -> None:
        """Append probability to sumtree. If sumtree is is_full of data, replace at oldest item.
        :param idx: index to append at
        :param priority: sample priority
        :return: None
        """
        if priority is not None:
            self._set_minmax(priority, priority)
        else:
            priority = self.max_priority

        self._update(idx, priority)
        self.entries = min(self.entries + 1, self.size)


if __name__ == '__main__':
    from collections import Counter
    from pprint import pprint

    size = 128
    a = np.linspace(start=0, stop=1, endpoint=False, num=size)
    a += a[1]
    sumtree = SumTree(len(a))
    sumtree.build(a)
    print(sumtree.tree)
    print(sumtree.tree[-sumtree.size:])
    dic = dict()

    print(f"\nSAMPLING")
    print(f"{'-'*50}")
    n_samples = 100000
    batch_size = 4
    for _ in range(n_samples):
        idxs, _ = sumtree.prioritized_sample(batch_size)
        priors = a[idxs]
        sumtree.update_weights(priors)
        for prior in priors:
            if prior in dic:
                dic[prior] += 100/n_samples
            else:
                dic[prior] = 100/n_samples

    pprint(Counter(dic))
