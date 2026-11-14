"""Shared size-measurement choices for PBM fitting and LOOP handoffs."""

FINAL_SIZE_TERM = "D10 D50 D90"
TREND_SIZE_TERM = "D10 D50 D90 trends"
SIZE_SAMPLE_OPTIONS = (
    ("Final D10 / D50 / D90", FINAL_SIZE_TERM),
    ("All recorded D10 / D50 / D90 trends", TREND_SIZE_TERM),
)
