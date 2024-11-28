"""
Track metrics with prometheus, so they can be scraped.
"""

from prometheus_client import Counter, Histogram, Gauge


total_requests = Counter(
    "invocation_total", "Total invocations", ["chute_id", "function", "status"]
)
request_duration = Histogram(
    "invocation_duration",
    "Invocation duration in seconds",
    ["chute_id", "function", "status"],
)
last_request_timestamp = Gauge(
    "invocation_last_timestamp",
    "Timestamp of the most recent invocation",
    ["chute_id", "function"],
)
