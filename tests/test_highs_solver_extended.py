"""
Extended tests for HiGHS Solver - covering edge cases and missing functionality
"""

import pytest
import tempfile
import os
from core.model import Model
from core.base import Variable, Constraint, Objective, VariableType, ConstraintType, SolutionStatus
from solvers.highs import HighsSolver


class TestBinaryAndIntegerVariables:
    """Test binary and integer variable handling"""

    def test_binary_variables(self):
        """Test model with binary variables"""
        model = Model(name="binary_test")

        # Binary variables
        x = Variable(name="x", var_type=VariableType.BINARY)
        y = Variable(name="y", var_type=VariableType.BINARY)
        model.add_variable(x)
        model.add_variable(y)

        # Maximize x + 2y subject to x + y <= 1
        model.objective = Objective(coefficients={"x": 1.0, "y": 2.0}, is_minimize=False)
        model.add_constraint(Constraint(
            name="c1",
            coefficients={"x": 1.0, "y": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=1.0
        ))

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()

        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        # Optimal should be x=0, y=1 (objective = 2)
        assert solution["x"] in [0.0, 1.0]
        assert solution["y"] in [0.0, 1.0]
        assert solution["x"] + solution["y"] <= 1.0 + 1e-6
        assert abs(solver.get_objective_value() - 2.0) < 1e-6

    def test_integer_variables(self):
        """Test model with integer variables"""
        model = Model(name="integer_test")

        x = Variable(name="x", var_type=VariableType.INTEGER, lower_bound=0.0, upper_bound=10.0)
        y = Variable(name="y", var_type=VariableType.INTEGER, lower_bound=0.0, upper_bound=10.0)
        model.add_variable(x)
        model.add_variable(y)

        # Minimize x + y subject to 2x + 3y >= 10
        model.objective = Objective(coefficients={"x": 1.0, "y": 1.0}, is_minimize=True)
        model.add_constraint(Constraint(
            name="c1",
            coefficients={"x": 2.0, "y": 3.0},
            constraint_type=ConstraintType.GEQ,
            rhs=10.0
        ))

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()

        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        # Solutions should be integers
        assert abs(solution["x"] - round(solution["x"])) < 1e-6
        assert abs(solution["y"] - round(solution["y"])) < 1e-6

    def test_mixed_variable_types(self):
        """Test model with continuous, integer, and binary variables"""
        model = Model(name="mixed_test")

        x_cont = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
        y_int = Variable(name="y", var_type=VariableType.INTEGER, lower_bound=0.0, upper_bound=10.0)
        z_bin = Variable(name="z", var_type=VariableType.BINARY)

        model.add_variable(x_cont)
        model.add_variable(y_int)
        model.add_variable(z_bin)

        model.objective = Objective(
            coefficients={"x": 1.0, "y": 2.0, "z": 3.0},
            is_minimize=True
        )
        model.add_constraint(Constraint(
            name="c1",
            coefficients={"x": 1.0, "y": 1.0, "z": 1.0},
            constraint_type=ConstraintType.GEQ,
            rhs=3.0
        ))

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()

        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        # y should be integer, z should be binary
        assert abs(solution["y"] - round(solution["y"])) < 1e-6
        assert solution["z"] in [0.0, 1.0]


class TestSolverConfiguration:
    """Test solver configuration methods"""

    def test_set_verbose_on(self):
        """Test enabling verbose output"""
        solver = HighsSolver()
        solver.set_verbose(True)
        assert solver.verbose is True

    def test_set_verbose_off(self):
        """Test disabling verbose output"""
        solver = HighsSolver()
        solver.set_verbose(False)
        assert solver.verbose is False

    def test_set_time_limit(self):
        """Test setting time limit"""
        solver = HighsSolver()
        solver.set_time_limit(10.0)
        # Just verify it doesn't crash - actual time limit behavior is HiGHS internal

    def test_solve_with_time_limit(self):
        """Test that time limit is respected (with a very short limit on complex problem)"""
        # Create a relatively complex MIP that might hit time limit
        model = Model(name="time_limit_test")

        for i in range(20):
            model.add_variable(Variable(
                name=f"x{i}",
                var_type=VariableType.INTEGER,
                lower_bound=0.0,
                upper_bound=100.0
            ))

        model.objective = Objective(
            coefficients={f"x{i}": float(i+1) for i in range(20)},
            is_minimize=False
        )

        # Add many constraints
        for i in range(50):
            model.add_constraint(Constraint(
                name=f"c{i}",
                coefficients={f"x{j}": float((i+j) % 3 + 1) for j in range(20)},
                constraint_type=ConstraintType.LEQ,
                rhs=100.0
            ))

        solver = HighsSolver()
        solver.load_model(model)
        solver.set_time_limit(0.001)  # Very short time limit
        status = solver.solve()

        # Should either solve optimally (fast problem) or hit time limit
        assert status in [SolutionStatus.OPTIMAL, SolutionStatus.TIME_LIMIT, SolutionStatus.UNKNOWN]


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_solve_empty_model(self):
        """Test solving model with no constraints"""
        model = Model(name="empty")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()

        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        # Should minimize to lower bound
        assert abs(solution["x"] - 0.0) < 1e-6

    def test_add_constraint_without_loading_model(self):
        """Test that adding constraint without loaded model raises error"""
        solver = HighsSolver()
        with pytest.raises(ValueError, match="Model must be loaded"):
            solver.add_constraint(Constraint(
                name="c1",
                coefficients={"x": 1.0},
                constraint_type=ConstraintType.LEQ,
                rhs=10.0
            ))

    def test_remove_constraint_without_loading_model(self):
        """Test that removing constraint without loaded model raises error"""
        solver = HighsSolver()
        with pytest.raises(ValueError, match="Model must be loaded"):
            solver.remove_constraint("c1")

    def test_set_constraint_active_without_loading_model(self):
        """Test that setting constraint active without loaded model raises error"""
        solver = HighsSolver()
        with pytest.raises(ValueError, match="Model must be loaded"):
            solver.set_constraint_active("c1", True)

    def test_activate_nonexistent_constraint(self):
        """Test activating constraint that was never added"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        solver = HighsSolver()
        solver.load_model(model)

        # Try to activate a constraint that doesn't exist - should raise
        initial_version = solver.synced_model_version
        with pytest.raises(ValueError):
            solver.set_constraint_active("nonexistent", True)
        # Version should not change since constraint doesn't exist
        assert solver.synced_model_version == initial_version

    def test_model_with_only_inactive_constraints(self):
        """Test model where all constraints are inactive"""
        model = Model(name="all_inactive")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        c1 = Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.GEQ,
            rhs=5.0,
            is_active=False
        )
        model.add_constraint(c1)

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()

        # Should solve as unconstrained (only bounds)
        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        assert abs(solution["x"] - 0.0) < 1e-6

    def test_constraint_with_no_variables(self):
        """Test constraint with empty coefficients"""
        model = Model(name="empty_constraint")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        # Constraint with no variables (always satisfied if rhs >= 0)
        model.add_constraint(Constraint(
            name="c1",
            coefficients={},
            constraint_type=ConstraintType.GEQ,
            rhs=-5.0
        ))

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()

        assert status == SolutionStatus.OPTIMAL

    def test_constraint_with_zero_coefficients(self):
        """Test constraint where all coefficients are zero"""
        model = Model(name="zero_coeff")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        model.add_constraint(Constraint(
            name="c1",
            coefficients={"x": 0.0},
            constraint_type=ConstraintType.GEQ,
            rhs=-1.0
        ))

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()

        assert status == SolutionStatus.OPTIMAL


class TestMaximization:
    """Test maximization problems"""

    def test_simple_maximization(self):
        """Test a simple maximization problem"""
        model = Model(name="max_test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
        model.add_variable(x)

        # Maximize x (should give x=10)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=False)
        model.add_constraint(Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=8.0
        ))

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()

        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        assert abs(solution["x"] - 8.0) < 1e-6
        assert abs(solver.get_objective_value() - 8.0) < 1e-6


class TestDualValues:
    """Test dual value extraction"""

    def test_get_dual_values(self):
        """Test getting dual values from optimal solution"""
        model = Model(name="dual_test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        y = Variable(name="y", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.add_variable(y)

        model.objective = Objective(coefficients={"x": 2.0, "y": 3.0}, is_minimize=True)

        model.add_constraint(Constraint(
            name="c1",
            coefficients={"x": 1.0, "y": 1.0},
            constraint_type=ConstraintType.GEQ,
            rhs=10.0
        ))
        model.add_constraint(Constraint(
            name="c2",
            coefficients={"x": 2.0, "y": 1.0},
            constraint_type=ConstraintType.GEQ,
            rhs=15.0
        ))

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()

        assert status == SolutionStatus.OPTIMAL

        duals = solver.get_dual_values()
        assert duals is not None
        assert "c1" in duals
        assert "c2" in duals
        # Dual values should be non-negative for minimization with >= constraints
        assert duals["c1"] >= -1e-6
        assert duals["c2"] >= -1e-6

    def test_dual_values_none_before_solve(self):
        """Test that dual values are None before solving"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        solver = HighsSolver()
        solver.load_model(model)

        # Before solving, dual values should be None
        assert solver.get_dual_values() is None


class TestIncrementalUpdateEdgeCases:
    """Test edge cases in incremental updates"""

    def test_full_reload_on_variable_addition(self):
        """Test that adding variables triggers full reload"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        solver = HighsSolver()
        solver.load_model(model)
        initial_version = solver.synced_model_version

        # Add a variable (should trigger full reload)
        y = Variable(name="y", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(y)

        solver.sync_model(model)

        # Version should be updated
        assert solver.synced_model_version > initial_version
        # Variable should be in index
        assert "y" in solver.var_indices

    def test_remove_nonexistent_constraint_incremental(self):
        """Test removing a constraint that doesn't exist in solver"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
        model.add_constraint(Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.GEQ,
            rhs=5.0
        ))

        solver = HighsSolver()
        solver.load_model(model)

        # Manually add a constraint to model but not to solver
        model.constraints["c2"] = Constraint(
            name="c2",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=10.0
        )

        # Now try to remove c2 (which isn't in solver)
        # This should not crash
        solver._remove_constraint_incremental("c2")

    def test_deactivate_nonexistent_constraint_incremental(self):
        """Test deactivating a constraint that doesn't exist in solver"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        solver = HighsSolver()
        solver.load_model(model)

        # Try to deactivate nonexistent constraint - should not crash
        solver._deactivate_constraint_incremental("nonexistent")
