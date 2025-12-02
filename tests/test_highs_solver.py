"""
Unit tests for HiGHS Solver implementation
"""

import pytest
from core.model import Model
from core.base import Variable, Constraint, Objective, VariableType, ConstraintType, SolutionStatus
from solvers.highs import HighsSolver


class TestHighsSolverBasics:
    """Test basic HiGHS solver functionality"""

    def test_solver_initialization(self):
        """Solver should initialize correctly"""
        solver = HighsSolver()
        assert solver.model is None
        assert solver.synced_model_version == -1	
        assert len(solver.var_indices) == 0
        assert len(solver.constraint_indices) == 0

    def test_supports_incremental_updates(self):
        """HiGHS solver should support incremental updates"""
        solver = HighsSolver()
        assert solver.supports_incremental_updates() is True

    def test_simple_feasible_lp(self):
        """Test solving a simple feasible LP"""
        # Model: minimize x + y
        #        subject to: x + y >= 5
        #                   x >= 0, y >= 0
        model = Model(name="simple_feasible")

        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        y = Variable(name="y", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.add_variable(y)

        model.objective = Objective(coefficients={"x": 1.0, "y": 1.0}, is_minimize=True)

        c1 = Constraint(
            name="c1",
            coefficients={"x": 1.0, "y": 1.0},
            constraint_type=ConstraintType.GEQ,
            rhs=5.0
        )
        model.add_constraint(c1)

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()

        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        assert solution is not None
        # Objective value should be 5.0 (x=5, y=0 or x=0, y=5, or any combination summing to 5)
        obj_value = solver.get_objective_value()
        assert obj_value is not None
        assert abs(obj_value - 5.0) < 1e-6

    def test_simple_infeasible_lp(self):
        """Test solving an infeasible LP"""
        # Model: minimize x
        #        subject to: x <= 5
        #                   x >= 10  (infeasible!)
        model = Model(name="simple_infeasible")

        x = Variable(name="x", var_type=VariableType.CONTINUOUS)
        model.add_variable(x)

        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        c1 = Constraint(name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.LEQ, rhs=5.0)
        c2 = Constraint(name="c2", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=10.0)
        model.add_constraint(c1)
        model.add_constraint(c2)

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()

        assert status == SolutionStatus.INFEASIBLE

    def test_two_variable_lp(self):
        """Test a standard 2-variable LP"""
        # maximize 3x + 2y
        # subject to: 2x + y <= 18
        #            2x + 3y <= 42
        #            3x + y <= 24
        #            x, y >= 0
        model = Model(name="two_var")

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

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()

        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        assert solution is not None

        # Optimal solution should be x=3, y=12 with objective value 33
        assert abs(solution["x"] - 3.0) < 1e-6
        assert abs(solution["y"] - 12.0) < 1e-6

        obj_value = solver.get_objective_value()
        assert obj_value is not None and abs(obj_value - 33.0) < 1e-6

    def test_equality_constraint(self):
        """Test model with equality constraints"""
        # minimize x + y
        # subject to: x + y = 10
        #            x, y >= 0
        model = Model(name="equality")

        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        y = Variable(name="y", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.add_variable(y)

        model.objective = Objective(coefficients={"x": 1.0, "y": 1.0}, is_minimize=True)

        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0, "y": 1.0}, constraint_type=ConstraintType.EQ, rhs=10.0
        ))

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()

        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        # x + y should equal 10
        assert abs(solution["x"] + solution["y"] - 10.0) < 1e-6

    def test_reset(self):
        """Test that reset clears solver state"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=5.0
        ))

        solver = HighsSolver()
        solver.load_model(model)
        solver.solve()

        solver.reset()

        assert solver.model is None
        assert solver.synced_model_version == -1
        assert len(solver.var_indices) == 0
        assert len(solver.constraint_indices) == 0

    def test_constraint_index_mapping(self):
        """Test that constraint indices are properly maintained"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        # Add multiple constraints
        model.add_constraint(Constraint(name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=1.0))
        model.add_constraint(Constraint(name="c2", coefficients={"x": 1.0}, constraint_type=ConstraintType.LEQ, rhs=10.0))
        model.add_constraint(Constraint(name="c3", coefficients={"x": 1.0}, constraint_type=ConstraintType.LEQ, rhs=8.0))

        solver = HighsSolver()
        solver.load_model(model)

        # Check that indices are assigned
        assert "c1" in solver.constraint_indices
        assert "c2" in solver.constraint_indices
        assert "c3" in solver.constraint_indices
        # Indices should be 0, 1, 2
        assert solver.constraint_indices["c1"] == 0
        assert solver.constraint_indices["c2"] == 1
        assert solver.constraint_indices["c3"] == 2

    def test_variable_index_mapping(self):
        """Test that variable indices are properly maintained"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        y = Variable(name="y", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        z = Variable(name="z", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.add_variable(y)
        model.add_variable(z)

        solver = HighsSolver()
        solver.load_model(model)

        # Variables are sorted alphabetically, so indices should be x=0, y=1, z=2
        assert solver.var_indices["x"] == 0
        assert solver.var_indices["y"] == 1
        assert solver.var_indices["z"] == 2
