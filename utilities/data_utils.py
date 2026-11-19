from pathlib import Path
import re
import time


def result_files(folder, filename):
    """Accept numbered results, including Windows' numbered copy suffixes.

    Examples: expttsd1, expttsd(1), expttsd_1 and expttsd1 (2).
    Descriptive backup suffixes still do not match.
    """
    pattern = re.compile(
        re.escape(filename) + r"(?:[ _-]*\d+|\s*\(\d+\))*(?:\s*-\s*Copy(?:\s*\(\d+\))?)*",
        re.IGNORECASE,
    )
    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in {".csv", ".mat", ".json", ".txt", ""}
        and pattern.fullmatch(p.stem)
    )


def counter(folder, filename):
    return len(result_files(folder, filename))

def file_finder(folder, filename):
    return [str(p) for p in result_files(folder, filename) if p.suffix.lower() == ".csv"]


class StableResultTracker:
    """Require a nonempty result to remain unchanged over a quiet interval.

    Producers should publish by atomic rename. Stability is only a fallback;
    it cannot prove completion for writers that pause longer than quiet_s.
    """
    def __init__(self, quiet_s=2.0, clock=time.monotonic):
        self.quiet_s, self.clock = quiet_s, clock
        self._observed = {}

    def files(self, folder, filename):
        now = self.clock()
        current, ready = {}, []
        for path in result_files(folder, filename):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            signature = (stat.st_size, stat.st_mtime_ns)
            old_signature, since = self._observed.get(path, (None, now))
            if signature != old_signature:
                since = now
            current[path] = (signature, since)
            if stat.st_size and now - since >= self.quiet_s:
                ready.append(path)
        self._observed = current
        return ready
