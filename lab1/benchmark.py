import subprocess
import re
import csv
import os
import statistics


# ============================================================
# SETTINGS
# ============================================================

# Number of times each configuration is tested
REPETITIONS = 3

# Fixed number of threads for Sheet 1
FIXED_THREADS = 8

# n values for Sheet 1
N_VALUES = [
    10_000_000,
    15_000_000,
    20_000_000,
    25_000_000,
    30_000_000,
    35_000_000,
    40_000_000,
    45_000_000,
    50_000_000,
    55_000_000,
    60_000_000,
    65_000_000,
    70_000_000,
    75_000_000,
    80_000_000,
    85_000_000,
    90_000_000,
    95_000_000,
    100_000_000,
    105_000_000,
    110_000_000,
    115_000_000,
    120_000_000,
    125_000_000,
    130_000_000,
    135_000_000,
    140_000_000,
    145_000_000,
    150_000_000,
    200_000_000
]


# Thread counts for Sheet 2
THREAD_COUNTS = [1, 2, 4, 8]

# Fixed n for Sheet 2
FIXED_N = 50_000_000


# ============================================================
# RUN A SINGLE PROGRAM
# ============================================================

def run_program(command):

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        print("\nERROR running:")
        print(" ".join(command))

        print("\nProgram output:")
        print(result.stdout)

        print("\nError:")
        print(result.stderr)

        return None

    # Task 1 and Task 3:
    #     Time taken: X seconds
    #
    # Task 2:
    #     Total time: X seconds

    match = re.search(
        r"(?:Time taken|Total time):\s*([0-9.]+)\s*seconds",
        result.stdout
    )

    if match is None:

        print("\nERROR: Could not find execution time.")

        print("Program output:")
        print(result.stdout)

        return None

    return float(match.group(1))

# ============================================================
# RUN PROGRAM MULTIPLE TIMES
# ============================================================

def benchmark(command, repetitions=REPETITIONS):

    times = []


    for run in range(1, repetitions + 1):

        print(
            f"      Run {run}/{repetitions}...",
            end=" ",
            flush=True
        )


        time_taken = run_program(command)


        if time_taken is None:
            return None


        times.append(time_taken)


        print(
            f"{time_taken:.6f} s"
        )


    average = statistics.mean(times)

    return {
        "times": times,
        "average": average,
        "minimum": min(times),
        "maximum": max(times)
    }


# ============================================================
# SHEET 1
#
# Vary n
# Fixed threads = 8
#
# Task 1 = Serial
# Task 2 = POSIX, 8 threads
# Task 3 = OpenMP, 8 threads
# ============================================================

def run_n_experiments():

    print("\n")
    print("=" * 60)
    print("EXPERIMENT 1: RUNTIME VS n")
    print("=" * 60)

    print(
        f"Fixed thread count: {FIXED_THREADS}"
    )

    print(
        f"Repetitions per configuration: {REPETITIONS}"
    )

    print()


    results = []


    for index, n in enumerate(N_VALUES, start=1):

        print("-" * 60)

        print(
            f"[{index}/{len(N_VALUES)}] "
            f"n = {n:,}"
        )

        print()


        # ----------------------------------------------------
        # Task 1
        # ----------------------------------------------------

        print("  Task 1 - Serial")

        serial = benchmark([
            "./task1",
            str(n)
        ])


        if serial is None:
            print("  Skipping this n because Task 1 failed.")
            continue


        print(
            f"      Average: {serial['average']:.6f} s"
        )

        print()


        # ----------------------------------------------------
        # Task 2
        # ----------------------------------------------------

        print(
            f"  Task 2 - POSIX "
            f"({FIXED_THREADS} threads)"
        )

        posix = benchmark([
            "./task2",
            str(n),
            str(FIXED_THREADS)
        ])


        if posix is None:
            print("  Skipping this n because Task 2 failed.")
            continue


        print(
            f"      Average: {posix['average']:.6f} s"
        )

        print()


        # ----------------------------------------------------
        # Task 3
        # ----------------------------------------------------

        print(
            f"  Task 3 - OpenMP "
            f"({FIXED_THREADS} threads)"
        )

        openmp = benchmark([
            "./task3",
            str(n),
            str(FIXED_THREADS)
        ])


        if openmp is None:
            print("  Skipping this n because Task 3 failed.")
            continue


        print(
            f"      Average: {openmp['average']:.6f} s"
        )

        print()


        # ----------------------------------------------------
        # Calculate speedups
        # ----------------------------------------------------

        posix_speedup = (
            serial["average"] /
            posix["average"]
        )

        openmp_speedup = (
            serial["average"] /
            openmp["average"]
        )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append({

            "n": n,

            "serial_time":
                serial["average"],

            "posix_time":
                posix["average"],

            "posix_speedup":
                posix_speedup,

            "openmp_time":
                openmp["average"],

            "openmp_speedup":
                openmp_speedup
        })


        print("  RESULTS")

        print(
            f"      Serial : "
            f"{serial['average']:.6f} s"
        )

        print(
            f"      POSIX  : "
            f"{posix['average']:.6f} s"
        )

        print(
            f"      POSIX speedup: "
            f"{posix_speedup:.3f}x"
        )

        print(
            f"      OpenMP : "
            f"{openmp['average']:.6f} s"
        )

        print(
            f"      OpenMP speedup: "
            f"{openmp_speedup:.3f}x"
        )

        print()


    return results


# ============================================================
# SHEET 2
#
# Fixed n
# Vary thread count
# ============================================================

def run_thread_experiments():

    print("\n")
    print("=" * 60)
    print("EXPERIMENT 2: RUNTIME VS THREAD COUNT")
    print("=" * 60)

    print(
        f"Fixed n: {FIXED_N:,}"
    )

    print(
        f"Thread counts: {THREAD_COUNTS}"
    )

    print(
        f"Repetitions per configuration: {REPETITIONS}"
    )

    print()


    results = []


    # --------------------------------------------------------
    # Serial baseline
    # --------------------------------------------------------

    print("-" * 60)

    print(
        f"Serial baseline "
        f"(n = {FIXED_N:,})"
    )

    print()


    serial = benchmark([
        "./task1",
        str(FIXED_N)
    ])


    if serial is None:
        print("ERROR: Serial benchmark failed.")

        return []


    serial_average = serial["average"]


    print(
        f"Serial average: "
        f"{serial_average:.6f} s"
    )

    print()


    # --------------------------------------------------------
    # Thread experiments
    # --------------------------------------------------------

    for index, threads in enumerate(
        THREAD_COUNTS,
        start=1
    ):

        print("-" * 60)

        print(
            f"[{index}/{len(THREAD_COUNTS)}] "
            f"{threads} thread(s)"
        )

        print()


        # ----------------------------------------------------
        # POSIX
        # ----------------------------------------------------

        print(
            f"  Task 2 - POSIX "
            f"({threads} threads)"
        )

        posix = benchmark([
            "./task2",
            str(FIXED_N),
            str(threads)
        ])


        if posix is None:
            print("  POSIX failed. Skipping.")
            continue


        print(
            f"      Average: "
            f"{posix['average']:.6f} s"
        )

        print()


        # ----------------------------------------------------
        # OpenMP
        # ----------------------------------------------------

        print(
            f"  Task 3 - OpenMP "
            f"({threads} threads)"
        )

        openmp = benchmark([
            "./task3",
            str(FIXED_N),
            str(threads)
        ])


        if openmp is None:
            print("  OpenMP failed. Skipping.")
            continue


        print(
            f"      Average: "
            f"{openmp['average']:.6f} s"
        )

        print()


        # ----------------------------------------------------
        # Speedups
        # ----------------------------------------------------

        posix_speedup = (
            serial_average /
            posix["average"]
        )

        openmp_speedup = (
            serial_average /
            openmp["average"]
        )


        results.append({

            "threads": threads,

            "serial_time":
                serial_average,

            "posix_time":
                posix["average"],

            "posix_speedup":
                posix_speedup,

            "openmp_time":
                openmp["average"],

            "openmp_speedup":
                openmp_speedup
        })


        print("  RESULTS")

        print(
            f"      Serial : "
            f"{serial_average:.6f} s"
        )

        print(
            f"      POSIX  : "
            f"{posix['average']:.6f} s"
        )

        print(
            f"      POSIX speedup: "
            f"{posix_speedup:.3f}x"
        )

        print(
            f"      OpenMP : "
            f"{openmp['average']:.6f} s"
        )

        print(
            f"      OpenMP speedup: "
            f"{openmp_speedup:.3f}x"
        )

        print()


    return results


# ============================================================
# SAVE SHEET 1 RESULTS
# ============================================================

def save_n_results(results):

    filename = "results_vs_n.csv"


    with open(
        filename,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)


        writer.writerow([
            "size of n",
            "serial time (s)",
            "POSIX time (s)",
            "POSIX speedup",
            "OpenMP time (s)",
            "OpenMP speedup"
        ])


        for row in results:

            writer.writerow([
                row["n"],
                f"{row['serial_time']:.6f}",
                f"{row['posix_time']:.6f}",
                f"{row['posix_speedup']:.6f}",
                f"{row['openmp_time']:.6f}",
                f"{row['openmp_speedup']:.6f}"
            ])


    print(
        f"Saved: {filename}"
    )


# ============================================================
# SAVE SHEET 2 RESULTS
# ============================================================

def save_thread_results(results):

    filename = "results_vs_threads.csv"


    with open(
        filename,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)


        writer.writerow([
            "number of threads",
            "serial time (s)",
            "POSIX time (s)",
            "POSIX speedup",
            "OpenMP time (s)",
            "OpenMP speedup"
        ])


        for row in results:

            writer.writerow([
                row["threads"],
                f"{row['serial_time']:.6f}",
                f"{row['posix_time']:.6f}",
                f"{row['posix_speedup']:.6f}",
                f"{row['openmp_time']:.6f}",
                f"{row['openmp_speedup']:.6f}"
            ])


    print(
        f"Saved: {filename}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("FIT3143 LAB 1 BENCHMARK")
    print("=" * 60)

    print()
    print(
        "This will automatically run Task 1, Task 2 and Task 3."
    )

    print(
        f"Each configuration will run "
        f"{REPETITIONS} times."
    )

    print()


    # Check executables exist
    programs = [
        "./task1",
        "./task2",
        "./task3"
    ]


    for program in programs:

        if not os.path.exists(program):

            print(
                f"ERROR: {program} not found."
            )

            print(
                "Make sure you compiled all three programs "
                "and are running benchmark.py from the "
                "correct folder."
            )

            exit(1)


    # --------------------------------------------------------
    # Experiment 1
    # --------------------------------------------------------

    n_results = run_n_experiments()

    save_n_results(n_results)


    # --------------------------------------------------------
    # Experiment 2
    # --------------------------------------------------------

    thread_results = run_thread_experiments()

    save_thread_results(thread_results)


    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    print()

    print(
        "Created:"
    )

    print(
        "  results_vs_n.csv"
    )

    print(
        "  results_vs_threads.csv"
    )

    print()

    print(
        "You can now copy the CSV values into Google Sheets."
    )

    print()