from core.problem import make_tfim_1d
from classical_solver import solve

problem = make_tfim_1d(n_sites=6, j=1.0, h=1.2)
result = solve(problem)

print(result.ground_energy)
print(result.energy_per_site)
print(result.expectation_values)
print(result.correlations)
print(result.entanglement)
print(result.comparison_view())