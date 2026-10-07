import time
import math
import mpmath as mp

from flint import fmpz_mat


def normalize(c):
    c = [int(x) for x in c]

    g = 0
    for x in c:
        g = math.gcd(g, abs(x))

    if g:
        c = [x // g for x in c]

    for x in c:
        if x:
            if x < 0:
                c = [-v for v in c]
            break

    return tuple(c)


def residual(c, values):
    s = mp.mpf("0")

    for a, x in zip(c, values):
        s += a * x

    return abs(s)


def build_lattice(values, Q):

    n = len(values)

    rows = []

    for i, x in enumerate(values):

        row = [0] * (n + 1)

        row[i] = 1
        row[-1] = int(mp.nint(Q * x))

        rows.append(row)

    return fmpz_mat(rows)


def extract_candidates(T, n, maxcoeff):

    result = []

    for i in range(T.nrows()):

        c = [
            int(T[i, j])
            for j in range(n)
        ]

        if not any(c):
            continue

        c = normalize(c)

        if max(abs(x) for x in c) <= maxcoeff:
            result.append(c)

    return list(dict.fromkeys(result))


def aps(values, maxcoeff=100000, precision=100, verbose=True):
    old_dps = mp.mp.dps
    values = [mp.mpf(x) for x in values]
    n = len(values)

    levels = [24]
    while levels[-1] < precision:
        nxt = int(levels[-1] * 1.7)
        if nxt > precision:
            nxt = precision
        if nxt == levels[-1]:
            break
        levels.append(nxt)

    history = {}

    known = set()

    for digits in levels:
        mp.mp.dps = digits

        start = time.perf_counter()

        Q = 10 ** max(10, digits - 4)

        B = build_lattice(values, Q)

        R, T = B.lll(
            transform=True,
            delta=0.75
        )

        current = extract_candidates(
            T,
            n,
            maxcoeff
        )

        short = min(R.nrows(), max(16, n * 2))

        for i in range(short):
            c = [int(R[i, j]) for j in range(n)]

            if not any(c):
                continue

            c = normalize(c)

            if max(abs(x) for x in c) <= maxcoeff:
                current.append(c)

        current = list(dict.fromkeys(current))

        for c in current:
            r = residual(c, values)

            if c not in history:
                history[c] = []

            history[c].append((digits, r))
            known.add(c)


        ranked = []

        for c in known:
            h = history[c]

            if not h:
                continue

            last_r = h[-1][1]

            if last_r == 0:
                score = mp.mpf("1000000")
            else:
                score = -mp.log10(last_r)

            if len(h) >= 2:
                old_r = h[-2][1]

                if old_r > 0 and last_r > 0:
                    improvement = mp.log10(old_r) - mp.log10(last_r)

                    expected = h[-1][0] - h[-2][0]

                    stability = improvement / max(1, expected)

                    score += 50 * stability
            else:
                stability = 0

            ranked.append((score, c, last_r))

        ranked.sort(
            key=lambda x: x[0],
            reverse=True
        )

        elapsed = time.perf_counter() - start

        best = None
        best_r = None

        if ranked:
            best = ranked[0][1]
            best_r = ranked[0][2]

        if verbose:
            print(
                "APS precision=%d time=%.6f candidates=%d "
                "best=%s residual=%s"
                % (
                    digits,
                    elapsed,
                    len(ranked),
                    best,
                    mp.nstr(best_r, 20)
                    if best_r is not None
                    else "none"
                )
            )

        if digits == precision and ranked:
            mp.mp.dps = precision + 60

            verify_count = min(
                len(ranked),
                max(8, n)
            )

            verified = []

            for i in range(verify_count):
                c = ranked[i][1]
                r = residual(c, values)

                verified.append((r, c))

            verified.sort(key=lambda x: x[0])

            final_r, final_c = verified[0]

            threshold = mp.power(
                10,
                -(precision - 20)
            )

            if final_r < threshold:
                if verbose:
                    print()
                    print("FOUND:", final_c)
                    print(
                        "FINAL RESIDUAL:",
                        mp.nstr(final_r, 120)
                    )

                mp.mp.dps = old_dps
                return final_c

    mp.mp.dps = old_dps

    best = None
    best_score = None

    for c, h in history.items():
        if not h:
            continue

        r = h[-1][1]

        if r == 0:
            return c

        score = -mp.log10(r)

        if len(h) >= 2:
            r0 = h[-2][1]

            if r0 > 0:
                improvement = mp.log10(r0) - mp.log10(r)
                score += 50 * improvement

        if best_score is None or score > best_score:
            best_score = score
            best = c

    return best


def test():

    mp.mp.dps = 160

    true = (
        4,
        -7,
        2,
        -9,
        1
    )

    x0 = mp.pi
    x1 = mp.e
    x2 = mp.sqrt(2)
    x3 = mp.log(2)

    x4 = -(
        true[0] * x0 +
        true[1] * x1 +
        true[2] * x2 +
        true[3] * x3
    )

    values = [
        x0,
        x1,
        x2,
        x3,
        x4
    ]

    print("Test results:")
    print()
    print("true:", true)
    print()

    t = time.perf_counter()

    result = aps(
        values,
        maxcoeff=100,
        precision=100,
        verbose=True
    )

    elapsed = (
        time.perf_counter()
        - t
    )

    print()
    print("result:", result)
    print(
        "t = %.6f sec"
        % elapsed
    )

    if result:

        mp.mp.dps = 160

        print(
            "Residual:",
            mp.nstr(
                residual(
                    result,
                    values
                ),
                120
            )
        )


if __name__ == "__main__":
    test()