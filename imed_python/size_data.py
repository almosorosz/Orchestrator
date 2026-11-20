"""Shared size-measurement choices for PBM fitting and LOOP handoffs."""

FINAL_SIZE_TERM = "D10 D50 D90"
TREND_SIZE_TERM = "D10 D50 D90 trends"
SIZE_SAMPLE_OPTIONS = (
    ("Final D10 / D50 / D90", FINAL_SIZE_TERM),
    ("All recorded D10 / D50 / D90 trends", TREND_SIZE_TERM),
)

MAX_FINAL_SIZE_SAMPLES = 2**31 - 1
FINAL_SIZE_SAMPLE_TOOLTIP = (
    "Average the last N valid D10 / D50 / D90 triplets in the experiment period. "
    "Use all available valid triplets if fewer than N were recorded. Set 1 to use the last valid triplet."
)


def validate_final_size_samples(value):
    try:
        count = int(value) if isinstance(value, str) else value
    except ValueError:
        count = None
    if type(count) is not int or not 1 <= count <= MAX_FINAL_SIZE_SAMPLES:
        raise ValueError("Enter a positive whole number of final size samples to average.")
    return count
