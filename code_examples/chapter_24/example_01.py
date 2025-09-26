"""
Chapter 24 - Example 1
Extracted from Healthcare AI Implementation Guide
"""

import pulp

def solve_resource_allocation(resources, demands, capacities, costs):
    """
    Solves a simple resource allocation problem using linear programming.

    Args:
        resources (list): List of resource names (e.g., [\'Doctor\', \'Nurse\', \'Technician\']).
        demands (dict): Dictionary of department demands, e.g.,
                        {\'Emergency_Dept\': {\'Doctor\': 20, \'Nurse\': 40}}.
        capacities (dict): Dictionary of total available capacity for each resource (e.g., {\'Doctor\': 50, \'Nurse\': 150}).
        costs (dict): Dictionary of cost per unit of resource (e.g., {\'Doctor\': 100, \'Nurse\': 50}).

    Returns:
        dict: A dictionary containing the optimal allocation and total cost, or None if no solution.
    """
    prob = pulp.LpProblem("Healthcare_Resource_Allocation", pulp.LpMinimize)

    departments = list(demands.keys())
    staff_types = resources

    \# Decision variables: x[d][s] = hours of staff_type \'s\' allocated to department \'d\'
    x = pulp.LpVariable.dicts("staff_allocation", (departments, staff_types), lowBound=0, cat=\'Continuous\')

    \# Objective function: Minimize total cost
    prob += pulp.lpSum(x[d][s] * costs[s] for d in departments for s in staff_types), "Total Cost"

    \# Constraints:
    \# 1. Meet demand for each staff type in each department
    for d in departments:
        for s in staff_types:
            prob += x[d][s] >= demands[d].get(s, 0), f"Demand_for_{s}_in_{d}"

    \# 2. Do not exceed total capacity for each staff type
    for s in staff_types:
        prob += pulp.lpSum(x[d][s] for d in departments) <= capacities[s], f"Capacity_of_{s}"

    \# Solve the problem
    prob.solve()

    if pulp.LpStatus[prob.status] == \'Optimal\':
        allocation_result = {}
        for d in departments:
            allocation_result[d] = {}
            for s in staff_types:
                allocation_result[d][s] = x[d][s].varValue

        total_cost = pulp.value(prob.objective)
        return {"optimal_allocation": allocation_result, "total_cost": total_cost}
    else:
        return None

if __name__ == "__main__":
    \# Example Usage:
    resources_available = [\'Doctor\', \'Nurse\', \'Technician\']
    department_demands = {
        \'Emergency_Dept\': {\'Doctor\': 20, \'Nurse\': 40, \'Technician\': 10},
        \'ICU\': {\'Doctor\': 15, \'Nurse\': 30, \'Technician\': 5},
        \'General_Ward\': {\'Doctor\': 10, \'Nurse\': 60, \'Technician\': 15}
    }
    resource_capacities = {
        \'Doctor\': 50,  \# Total available doctor hours
        \'Nurse\': 150,  \# Total available nurse hours
        \'Technician\': 30 \# Total available technician hours
    }
    resource_costs = {
        \'Doctor\': 100,  \# Cost per doctor hour
        \'Nurse\': 50,    \# Cost per nurse hour
        \'Technician\': 40 \# Cost per technician hour
    }

    result = solve_resource_allocation(resources_available, department_demands, resource_capacities, resource_costs)

    if result:
        print("Optimal Allocation:")
        for dept, allocations in result[\'optimal_allocation\'].items():
            print(f"  {dept}:")
            for res, amount in allocations.items():
                print(f"    {res}: {amount:.2f} hours")
        print(f"Total Minimum Cost: ${result[\'total_cost\']:.2f}")
    else:
        print("No optimal solution found.")

    \# Example with insufficient capacity
    print("\n--- Example with Insufficient Capacity ---")
    resource_capacities_low = {
        \'Doctor\': 30,  \# Insufficient doctor hours
        \'Nurse\': 100,  \# Insufficient nurse hours
        \'Technician\': 20
    }
    result_low_capacity = solve_resource_allocation(resources_available, department_demands, resource_capacities_low, resource_costs)
    if result_low_capacity:
        print("Optimal Allocation:")
        for dept, allocations in result_low_capacity[\'optimal_allocation\'].items():
            print(f"  {dept}:")
            for res, amount in allocations.items():
                print(f"    {res}: {amount:.2f} hours")
        print(f"Total Minimum Cost: ${result_low_capacity[\'total_cost\']:.2f}")
    else:
        print("No optimal solution found due to insufficient capacity.")