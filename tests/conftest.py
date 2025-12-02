"""
Pytest configuration and fixtures for IIS-Finder tests
"""

import pytest
from core.model import Model
from core.base import Variable, Constraint, Objective, VariableType, ConstraintType
from solvers.highs import HighsSolver


@pytest.fixture
def simple_model():
    """Fixture providing a simple 1-variable model"""
    model = Model(name="simple")
    x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
    model.add_variable(x)
    model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
    return model


@pytest.fixture
def two_var_model():
    """Fixture providing a 2-variable model"""
    model = Model(name="two_var")
    x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
    y = Variable(name="y", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
    model.add_variable(x)
    model.add_variable(y)
    model.objective = Objective(coefficients={"x": 1.0, "y": 1.0}, is_minimize=True)
    return model


@pytest.fixture
def feasible_model():
    """Fixture providing a simple feasible model"""
    # minimize x + y
    # subject to: x + y >= 10, x >= 0, y >= 0
    # Optimal: x=0, y=10 or x=10, y=0, etc.
    model = Model(name="feasible")
    x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
    y = Variable(name="y", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
    model.add_variable(x)
    model.add_variable(y)
    model.objective = Objective(coefficients={"x": 1.0, "y": 1.0}, is_minimize=True)
    model.add_constraint(Constraint(
        name="c1",
        coefficients={"x": 1.0, "y": 1.0},
        constraint_type=ConstraintType.GEQ,
        rhs=10.0
    ))
    return model


@pytest.fixture
def infeasible_model():
    """Fixture providing a simple infeasible model"""
    # minimize x
    # subject to: x <= 5, x >= 10  (infeasible!)
    model = Model(name="infeasible")
    x = Variable(name="x", var_type=VariableType.CONTINUOUS)
    model.add_variable(x)
    model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
    model.add_constraint(Constraint(
        name="c1",
        coefficients={"x": 1.0},
        constraint_type=ConstraintType.LEQ,
        rhs=5.0
    ))
    model.add_constraint(Constraint(
        name="c2",
        coefficients={"x": 1.0},
        constraint_type=ConstraintType.GEQ,
        rhs=10.0
    ))
    return model


@pytest.fixture
def highs_solver():
    """Fixture providing a fresh HiGHS solver instance"""
    return HighsSolver()


@pytest.fixture
def loaded_solver(feasible_model):
    """Fixture providing a solver with a loaded model"""
    solver = HighsSolver()
    solver.load_model(feasible_model)
    return solver


@pytest.fixture
def multi_constraint_model():
    """Fixture providing a model with multiple constraints for testing toggles"""
    model = Model(name="multi")
    x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=100.0)
    model.add_variable(x)
    model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

    # Add several constraints with different RHS values
    model.add_constraint(Constraint(
        name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=10.0
    ))
    model.add_constraint(Constraint(
        name="c2", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=20.0
    ))
    model.add_constraint(Constraint(
        name="c3", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=30.0
    ))

    return model


@pytest.fixture
def standard_lp_model():
    """Fixture providing a standard LP problem (maximization)"""
    # maximize 3x + 2y
    # subject to: 2x + y <= 18
    #            2x + 3y <= 42
    #            3x + y <= 24
    #            x, y >= 0
    # Optimal: x=3, y=12, obj=33
    model = Model(name="standard_lp")

    x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
    y = Variable(name="y", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
    model.add_variable(x)
    model.add_variable(y)

    model.objective = Objective(coefficients={"x": 3.0, "y": 2.0}, is_minimize=False)

    model.add_constraint(Constraint(
        name="c1", coefficients={"x": 2.0, "y": 1.0}, constraint_type=ConstraintType.LEQ, rhs=18.0
    ))
    model.add_constraint(Constraint(
        name="c2", coefficients={"x": 2.0, "y": 3.0}, constraint_type=ConstraintType.LEQ, rhs=42.0
    ))
    model.add_constraint(Constraint(
        name="c3", coefficients={"x": 3.0, "y": 1.0}, constraint_type=ConstraintType.LEQ, rhs=24.0
    ))

    return model


# Helper functions that can be used across tests

def assert_solution_close(solver, expected, tolerance=1e-6):
    """Helper to assert solution values are close to expected"""
    solution = solver.get_solution()
    for var_name, expected_val in expected.items():
        assert var_name in solution, f"Variable {var_name} not in solution"
        actual_val = solution[var_name]
        assert abs(actual_val - expected_val) < tolerance, \
            f"Variable {var_name}: expected {expected_val}, got {actual_val}"


def assert_objective_close(solver, expected_obj, tolerance=1e-6):
    """Helper to assert objective value is close to expected"""
    actual_obj = solver.get_objective_value()
    assert actual_obj is not None, "Objective value is None"
    assert abs(actual_obj - expected_obj) < tolerance, \
        f"Objective: expected {expected_obj}, got {actual_obj}"


# Make helpers available for import
__all__ = [
    'simple_model',
    'two_var_model',
    'feasible_model',
    'infeasible_model',
    'highs_solver',
    'loaded_solver',
    'multi_constraint_model',
    'standard_lp_model',
    'assert_solution_close',
    'assert_objective_close',
]
