"""Half-open reporting intervals with consistent floating-point boundaries."""

from bisect import bisect_right
import math


class TimeBins:
    """Index against actual edges, never rounded timestamp/width quotients.

    The final edge is exclusive, including when the last interval is shorter
    than the reporting cadence. Validate edges once; lookup is O(log bins).
    """

    def __init__(self, edges):
        self.edges = tuple(edges)
        if (len(self.edges) < 2 or not all(math.isfinite(t) for t in self.edges)
                or any(a >= b for a, b in zip(self.edges, self.edges[1:]))):
            raise ValueError("Reporting edges must be finite and strictly increasing")

    def index(self, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError("Reporting timestamp must be finite")
        if timestamp < self.edges[0] or timestamp >= self.edges[-1]:
            return None
        return bisect_right(self.edges, timestamp) - 1
