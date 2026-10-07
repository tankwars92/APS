import time
import random
import math
import statistics
import mpmath as mp

from aps import aps


PRECISION = 80
REPEATS = 7
SEED = 20261007


def make_case(n, coeff_max, rng):
    coeff = [
        rng.randint(-coeff_max, coeff_max)
        for _ in range(n - 1)
    ]

    coeff = [
        x if x != 0 else 1
        for x in coeff
    ]

    coeff.append(1)

    base = [
        mp.mpf(rng.randint(1, 1000000))
        / rng.randint(1, 1000)
        + mp.sqrt(rng.randint(2, 1000))
        for _ in range(n - 1)
    ]

    last = -sum(
        coeff[i] * base[i]
        for i in range(n - 1)
    )

    return tuple(coeff), tuple(base + [last])


def norm(c):
    if c is None:
        return None

    c = tuple(int(x) for x in c)

    g = 0

    for x in c:
        g = math.gcd(g, abs(x))

    if g:
        c = tuple(x // g for x in c)

    for x in c:
        if x:
            if x < 0:
                c = tuple(-v for v in c)
            break

    return c


def run_aps(values, coeff_max):
    t0 = time.perf_counter()

    result = aps(
        values,
        maxcoeff=coeff_max * 20,
        precision=PRECISION,
        verbose=False
    )

    elapsed = time.perf_counter() - t0

    return result, elapsed


def run_pslq(values, coeff_max):
    t0 = time.perf_counter()

    result = mp.pslq(
        mp.matrix(values),
        tol=mp.power(10, -PRECISION),
        maxcoeff=coeff_max * 20,
        maxsteps=10000
    )

    elapsed = time.perf_counter() - t0

    return result, elapsed


def benchmark_case(n, coeff_max, rng):

    true, values = make_case(
        n,
        coeff_max,
        rng
    )

    run_aps(values, coeff_max)
    run_pslq(values, coeff_max)

    aps_times = []
    pslq_times = []

    aps_result = None
    pslq_result = None

    for i in range(REPEATS):

        if i % 2 == 0:

            aps_result, t = run_aps(
                values,
                coeff_max
            )
            aps_times.append(t)

            pslq_result, t = run_pslq(
                values,
                coeff_max
            )
            pslq_times.append(t)

        else:

            pslq_result, t = run_pslq(
                values,
                coeff_max
            )
            pslq_times.append(t)

            aps_result, t = run_aps(
                values,
                coeff_max
            )
            aps_times.append(t)

    aps_time = statistics.median(aps_times)
    pslq_time = statistics.median(pslq_times)

    aps_ok = (
        norm(aps_result) ==
        norm(true)
    )

    pslq_ok = (
        norm(pslq_result) ==
        norm(true)
    )

    if aps_time < pslq_time:
        winner = "APS"
    else:
        winner = "PSLQ"

    speedup = (
        pslq_time / aps_time
        if aps_time > 0
        else 0
    )

    return {
        "n": n,
        "coeff": coeff_max,
        "aps": aps_time,
        "pslq": pslq_time,
        "speedup": speedup,
        "winner": winner,
        "aps_ok": aps_ok,
        "pslq_ok": pslq_ok,
    }


def main():

    mp.mp.dps = PRECISION + 30

    rng = random.Random(SEED)

    cases = [
        (4, 10),
        (6, 10),
        (8, 10),
        (10, 10),
        (12, 10),
        (15, 10),
        (20, 10),
        (25, 10),
        (30, 10),

        (6, 100),
        (10, 100),
        (15, 100),
        (20, 100),

        (6, 1000),
        (10, 1000),
        (15, 1000),
    ]

    print()
    print("APS vs PSLQ")
    print()

    print(
        "%-6s %-8s %-13s %-13s %-12s %-8s %-8s"
        % (
            "DIM",
            "COEFF",
            "APS(sec)",
            "PSLQ(sec)",
            "PSLQ/APS",
            "WINNER",
            "OK"
        )
    )

    print("-" * 82)

    results = []

    for n, coeff_max in cases:

        try:
            result = benchmark_case(
                n,
                coeff_max,
                rng
            )

        except Exception as e:

            print(
                "%-6d %-8d ERROR: %s"
                % (
                    n,
                    coeff_max,
                    repr(e)
                )
            )

            continue

        results.append(result)

        ok = (
            "YES"
            if result["aps_ok"] and
               result["pslq_ok"]
            else
            "NO"
        )

        print(
            "%-6d %-8d %-13.6f %-13.6f %-12.2fx %-8s %-8s"
            % (
                result["n"],
                result["coeff"],
                result["aps"],
                result["pslq"],
                result["speedup"],
                result["winner"],
                ok
            )
        )


if __name__ == "__main__":
    main()