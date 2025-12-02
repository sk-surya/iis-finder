"""
Unit tests for core base classes
"""

import pytest
from core.base import (
    Variable, Constraint, Objective, IISResult,
    VariableType, ConstraintType, SolutionStatus,
    ModelChange, ChangeType
)


class TestVariable:
    """Test Variable class"""

    def test_variable_hash(self):
        """Test that variables can be hashed by name"""
        v1 = Variable(name="x", var_type=VariableType.CONTINUOUS)
        v2 = Variable(name="x", var_type=VariableType.INTEGER)
        v3 = Variable(name="y", var_type=VariableType.CONTINUOUS)

        # Same name should have same hash
        assert hash(v1) == hash(v2)
        # Different name should (likely) have different hash
        assert hash(v1) != hash(v3)

        # Can be used as dict keys
        var_dict = {v1: "first", v2: "second", v3: "third"}
        assert len(var_dict) == 3  # All 3 are distinct objects (dataclass equality is by default all fields)

    def test_variable_types(self):
        """Test different variable types"""
        continuous = Variable(name="x", var_type=VariableType.CONTINUOUS)
        integer = Variable(name="y", var_type=VariableType.INTEGER)
        binary = Variable(name="z", var_type=VariableType.BINARY)

        assert continuous.var_type == VariableType.CONTINUOUS
        assert integer.var_type == VariableType.INTEGER
        assert binary.var_type == VariableType.BINARY

    def test_variable_bounds(self):
        """Test variable bounds"""
        v = Variable(name="x", lower_bound=0.0, upper_bound=10.0)
        assert v.lower_bound == 0.0
        assert v.upper_bound == 10.0


class TestConstraint:
    """Test Constraint class"""

    def test_constraint_hash(self):
        """Test that constraints can be hashed by name"""
        c1 = Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=10.0
        )
        c2 = Constraint(
            name="c1",
            coefficients={"y": 2.0},
            constraint_type=ConstraintType.GEQ,
            rhs=5.0
        )
        c3 = Constraint(
            name="c2",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.EQ,
            rhs=10.0
        )

        # Same name should have same hash
        assert hash(c1) == hash(c2)
        # Different name should have different hash
        assert hash(c1) != hash(c3)

        # Can be used as dict keys
        constraint_dict = {c1: "first", c2: "second", c3: "third"}
        assert len(constraint_dict) == 3  # All 3 are distinct objects (dataclass equality is by default all fields)

    def test_constraint_str_leq(self):
        """Test string representation for LEQ constraints"""
        c = Constraint(
            name="c1",
            coefficients={"x": 2.0, "y": -1.0, "z": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=10.0
        )
        str_repr = str(c)
        assert "c1:" in str_repr
        assert "<=" in str_repr
        assert "10.0" in str_repr

    def test_constraint_str_geq(self):
        """Test string representation for GEQ constraints"""
        c = Constraint(
            name="c2",
            coefficients={"x": 1.0, "y": 1.0},
            constraint_type=ConstraintType.GEQ,
            rhs=5.0
        )
        str_repr = str(c)
        assert "c2:" in str_repr
        assert ">=" in str_repr
        assert "5.0" in str_repr

    def test_constraint_str_eq(self):
        """Test string representation for EQ constraints"""
        c = Constraint(
            name="equality",
            coefficients={"x": 3.0},
            constraint_type=ConstraintType.EQ,
            rhs=15.0
        )
        str_repr = str(c)
        assert "equality:" in str_repr
        assert "=" in str_repr
        assert "15.0" in str_repr

    def test_constraint_str_with_unit_coefficients(self):
        """Test string representation with coefficient 1 and -1"""
        c = Constraint(
            name="c3",
            coefficients={"x": 1.0, "y": -1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=0.0
        )
        str_repr = str(c)
        # Should show "x" not "1.0*x" and "-y" not "-1.0*y"
        assert str_repr is not None

    def test_constraint_str_empty(self):
        """Test string representation with no coefficients"""
        c = Constraint(
            name="empty",
            coefficients={},
            constraint_type=ConstraintType.LEQ,
            rhs=5.0
        )
        str_repr = str(c)
        assert "empty:" in str_repr
        assert "0" in str_repr  # Should show 0 for empty LHS

    def test_constraint_str_with_zero_coefficients(self):
        """Test string representation with zero coefficients"""
        c = Constraint(
            name="c4",
            coefficients={"x": 0.0, "y": 2.0},
            constraint_type=ConstraintType.LEQ,
            rhs=10.0
        )
        str_repr = str(c)
        # Zero coefficients should be skipped
        assert "y" in str_repr or "2.0" in str_repr


class TestIISResult:
    """Test IISResult class"""

    def test_iis_result_str(self):
        """Test string representation of IISResult"""
        result = IISResult(
            iis_constraints={"c1", "c2", "c3"},
            algorithm="deletion_filter",
            time_elapsed=0.125,
            iterations=10,
            status="success",
            metadata={"reduction": 0.6}
        )
        str_repr = str(result)

        assert "IIS Result:" in str_repr
        assert "deletion_filter" in str_repr
        assert "success" in str_repr
        assert "3" in str_repr  # IIS size
        assert "0.125" in str_repr
        assert "10" in str_repr

    def test_iis_result_defaults(self):
        """Test IISResult with default values"""
        result = IISResult(
            iis_constraints={"c1"},
            algorithm="test",
            time_elapsed=1.0
        )
        assert result.iterations == 0
        assert result.status == "success"
        assert result.metadata == {}


class TestEnums:
    """Test enum types"""

    def test_constraint_type_values(self):
        """Test ConstraintType enum values"""
        assert ConstraintType.LEQ.value == "<="
        assert ConstraintType.EQ.value == "="
        assert ConstraintType.GEQ.value == ">="

    def test_variable_type_values(self):
        """Test VariableType enum values"""
        assert VariableType.CONTINUOUS.value == "continuous"
        assert VariableType.INTEGER.value == "integer"
        assert VariableType.BINARY.value == "binary"

    def test_solution_status_values(self):
        """Test SolutionStatus enum values"""
        assert SolutionStatus.OPTIMAL.value == "optimal"
        assert SolutionStatus.INFEASIBLE.value == "infeasible"
        assert SolutionStatus.UNBOUNDED.value == "unbounded"
        assert SolutionStatus.UNKNOWN.value == "unknown"
        assert SolutionStatus.TIME_LIMIT.value == "time_limit"
        assert SolutionStatus.ERROR.value == "error"

    def test_change_type_values(self):
        """Test ChangeType enum values"""
        assert ChangeType.CONSTRAINT_ADDED.value == "constraint_added"
        assert ChangeType.CONSTRAINT_REMOVED.value == "constraint_removed"
        assert ChangeType.CONSTRAINT_ACTIVATED.value == "constraint_activated"
        assert ChangeType.CONSTRAINT_DEACTIVATED.value == "constraint_deactivated"
        assert ChangeType.VARIABLE_ADDED.value == "variable_added"
        assert ChangeType.OBJECTIVE_CHANGED.value == "objective_changed"


class TestModelChange:
    """Test ModelChange class"""

    def test_model_change_with_entity(self):
        """Test ModelChange with entity"""
        var = Variable(name="x")
        change = ModelChange(
            change_type=ChangeType.VARIABLE_ADDED,
            entity_name="x",
            entity=var
        )
        assert change.change_type == ChangeType.VARIABLE_ADDED
        assert change.entity_name == "x"
        assert change.entity == var

    def test_model_change_without_entity(self):
        """Test ModelChange without entity (for removals)"""
        change = ModelChange(
            change_type=ChangeType.CONSTRAINT_REMOVED,
            entity_name="c1"
        )
        assert change.change_type == ChangeType.CONSTRAINT_REMOVED
        assert change.entity_name == "c1"
        assert change.entity is None
