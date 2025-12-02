"""
Tests for IISAlgorithm base class
"""

import pytest
from core.base import IISAlgorithm, IISResult, Variable, Constraint, Objective
from core.base import VariableType, ConstraintType, SolutionStatus
from core.model import Model
from solvers.highs import HighsSolver


class DummyIISAlgorithm(IISAlgorithm):
    """Concrete implementation for testing"""

    def find_iis(self, model: Model) -> IISResult:
        """Dummy implementation that returns all constraints"""
        return IISResult(
            iis_constraints=set(model.constraints.keys()),
            algorithm="dummy",
            time_elapsed=0.0
        )


class TestIISAlgorithm:
    """Test IISAlgorithm base class functionality"""

    def test_algorithm_initialization(self):
        """Test that algorithm initializes with solver"""
        solver = HighsSolver()
        algorithm = DummyIISAlgorithm(solver)

        assert algorithm.solver == solver
        assert algorithm.model is None

    def test_verify_iis_valid(self):
        """Test verifying a valid IIS"""
        # Create an infeasible model: x <= 5 and x >= 10
        model = Model(name="infeasible")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        c1 = Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=5.0
        )
        c2 = Constraint(
            name="c2",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.GEQ,
            rhs=10.0
        )
        model.add_constraint(c1)
        model.add_constraint(c2)

        solver = HighsSolver()
        algorithm = DummyIISAlgorithm(solver)

        # Both constraints form a valid IIS
        iis_constraints = {"c1", "c2"}
        # iis_constraints = algorithm.find_iis(model).iis_constraints
        is_valid = algorithm.verify_iis(model, iis_constraints)

        assert is_valid is True

    def test_verify_iis_not_infeasible(self):
        """Test that verify_iis returns False if constraints are feasible"""
        # Create a feasible model
        model = Model(name="feasible")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        c1 = Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.GEQ,
            rhs=5.0
        )
        model.add_constraint(c1)

        solver = HighsSolver()
        algorithm = DummyIISAlgorithm(solver)

        # Single constraint is feasible, not an IIS
        iis_constraints = {"c1"}
        is_valid = algorithm.verify_iis(model, iis_constraints)

        assert is_valid is False

    def test_verify_iis_not_minimal(self):
        """Test that verify_iis returns False if IIS is not minimal"""
        # Create a model with redundant constraints
        model = Model(name="redundant")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        c1 = Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=5.0
        )
        c2 = Constraint(
            name="c2",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.GEQ,
            rhs=10.0
        )
        c3 = Constraint(
            name="c3",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.GEQ,
            rhs=8.0  # Redundant with c2
        )
        model.add_constraint(c1)
        model.add_constraint(c2)
        model.add_constraint(c3)

        solver = HighsSolver()
        algorithm = DummyIISAlgorithm(solver)

        # All three constraints together are infeasible,
        # but removing c3 still leaves it infeasible (c1 and c2 form the minimal IIS)
        iis_constraints = {"c1", "c2", "c3"}
        is_valid = algorithm.verify_iis(model, iis_constraints)

        assert is_valid is False

    def test_verify_iis_missing_constraint(self):
        """Test verifying IIS when a constraint name doesn't exist"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        c1 = Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=5.0
        )
        model.add_constraint(c1)

        solver = HighsSolver()
        algorithm = DummyIISAlgorithm(solver)

        # Try to verify with a nonexistent constraint
        iis_constraints = {"c1", "nonexistent"}
        # Should handle gracefully (skip nonexistent constraints)
        is_valid = algorithm.verify_iis(model, iis_constraints)

        # Should return False because c1 alone is feasible
        assert is_valid is False
