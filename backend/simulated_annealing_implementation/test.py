from tn_sa_exemple import compare, make_2x2_tn, make_erdos_renyi_tn, make_square_tn


def run_examples() -> None:
    print("Simulated Annealing for Tensor Network Contraction Ordering")
    print("Examples based on the original quimb script\n")

    experiments = [
        ("2x2 Square, chi=2", make_2x2_tn(2), 5_000, 3),
        ("3x3 Square, chi=2", make_square_tn(3, 2, seed=1), 20_000, 5),
        ("3x3 Square, chi=10", make_square_tn(3, 10, seed=1), 20_000, 5),
        (
            "Erdos-Renyi 10 nodes, chi=10",
            make_erdos_renyi_tn(10, p=0.8, chi=10, seed=42),
            30_000,
            5,
        ),
    ]

    for label, tn, sa_iterations, n_sa_runs in experiments:
        compare(
            tn=tn,
            label=label,
            sa_iterations=sa_iterations,
            n_sa_runs=n_sa_runs,
        )
#Debug to check that the code works on smaller networks
def run_debug():
    print("DEBUG MODE - lightweight tests\n")

    experiments = [
        ("2x2 Square, chi=2", make_2x2_tn(2), 500,5),
        ("2x2 Square, chi=3", make_2x2_tn(3), 500,5),
        ("Small ER graph", make_erdos_renyi_tn(7, p=0.5, chi=2, seed=0), 500,5),
    ]

    for label, tn, sa_iter, runs in experiments:
        compare(tn, label, sa_iter, runs)

def main() -> None:
    # run_examples() #To uncomment to run the actual code
    run_debug()


if __name__ == "__main__":
    main()