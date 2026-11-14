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

def corr_func(x,X):
    A = x[0] * X[0] + x[1]
    C = (A.iloc[0].values - X[0].iloc[0].values)**2
    C = (A.iloc[0].values - 0.07332960848175958)**2
    D = (A.iloc[-1].values - X[1].iloc[-1].values)**2

    return  C+D

def W_pred(X_new):
    mu_X = [151.74736042]
    sigma_X = [34.94415798]
    B = [[1.89453211, 3.1580302,  4.58156681]]
    A = [ 9.93866667, 20.08833333, 37.43866667]
    X_new_scaled = (X_new - mu_X) / sigma_X
    Y_scaled_pred_direct = X_new_scaled @ B  
    Y_pred_direct = Y_scaled_pred_direct + A 
    
    return Y_pred_direct



def L_pred(X_new):
    mu_X    = [201.78218686, 736.3733133 ]
    sigma_X = [ 91.934371,   239.33190586]
    B       = [[30.47983386, 56.41028485, 84.9595095 ], [32.95049616, 60.98284141, 91.84623527]]
    A       = [138.22833333, 290.32783333, 434.75716667]
    X_new_scaled = (X_new - mu_X) / sigma_X
    Y_scaled_pred_direct = X_new_scaled @ B  
    Y_pred_direct = Y_scaled_pred_direct + A 
    
    return Y_pred_direct

def smoother(LLL,T1,LX,k):
    import numpy as np
    T = [np.mean([ j for j in LLL.iloc[i:i+k,LX] ]) for i in range(len(LLL)-k)]
    t = np.linspace(min(T1),max(T1),len(T)).reshape(1,-1)
    return t[0], np.array(T).reshape(1,-1)[0]

def solub(T):
    return (0.000011181*(T**2) + 0.0004036447*T + 0.0368389481)

def final_value(T1, WWW_1,finaltime):
    import numpy as np
    mask = (T1 > max(T1) - finaltime)
    return np.mean(WWW_1[mask][:],0)

def SED(L,W):
    
    return ((6*L*(W**2))/3.141592653589793)**(1/3)

def has_existing_files(folder, filename):
    import os

    for root, dirs, files in os.walk(folder):
        if any(filename in file for file in files):
            return True

    return False
