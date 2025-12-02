"""
Unit tests for Model change tracking functionality
"""

import pytest
from core.model import Model
from core.base import Variable, Constraint, Objective, VariableType, ConstraintType, ChangeType


class TestModelChangeTracking:
    """Test that Model properly tracks changes"""

    def test_initial_version_is_zero(self):
        """Model should start at version 0"""
        model = Model(name="test")
        assert model.get_version() == 0
        assert len(model._changes) == 0

    def test_add_variable_increments_version(self):
        """Adding a variable should increment version and log change"""
        model = Model(name="test")
        var = Variable(name="x", var_type=VariableType.CONTINUOUS)

        model.add_variable(var)

        assert model.get_version() == 1
        assert len(model._changes) == 1
        assert model._changes[0].change_type == ChangeType.VARIABLE_ADDED
        assert model._changes[0].entity_name == "x"
        assert model._changes[0].entity == var

    def test_add_constraint_increments_version(self):
        """Adding a constraint should increment version and log change"""
        model = Model(name="test")
        constraint = Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=10.0
        )

        model.add_constraint(constraint)

        assert model.get_version() == 1
        assert len(model._changes) == 1
        assert model._changes[0].change_type == ChangeType.CONSTRAINT_ADDED
        assert model._changes[0].entity_name == "c1"
        assert model._changes[0].entity == constraint

    def test_remove_constraint_increments_version(self):
        """Removing a constraint should increment version and log change"""
        model = Model(name="test")
        constraint = Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=10.0
        )
        model.add_constraint(constraint)

        model.remove_constraint("c1")

        assert model.get_version() == 2
        assert len(model._changes) == 2
        assert model._changes[1].change_type == ChangeType.CONSTRAINT_REMOVED
        assert model._changes[1].entity_name == "c1"

    def test_deactivate_constraint_increments_version(self):
        """Deactivating a constraint should increment version and log change"""
        model = Model(name="test")
        constraint = Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=10.0
        )
        model.add_constraint(constraint)

        model.deactivate_constraint("c1")

        assert model.get_version() == 2
        assert len(model._changes) == 2
        assert model._changes[1].change_type == ChangeType.CONSTRAINT_DEACTIVATED
        assert model._changes[1].entity_name == "c1"
        assert model.constraints["c1"].is_active is False

    def test_activate_constraint_increments_version(self):
        """Activating a constraint should increment version and log change"""
        model = Model(name="test")
        constraint = Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=10.0
        )
        model.add_constraint(constraint)
        model.deactivate_constraint("c1")

        model.activate_constraint("c1")

        assert model.get_version() == 3
        assert len(model._changes) == 3
        assert model._changes[2].change_type == ChangeType.CONSTRAINT_ACTIVATED
        assert model._changes[2].entity_name == "c1"
        assert model.constraints["c1"].is_active is True

    def test_get_changes_since_version(self):
        """Should return only changes after a specific version"""
        model = Model(name="test")

        # Make several changes
        model.add_variable(Variable(name="x"))  # version 1
        model.add_variable(Variable(name="y"))  # version 2
        model.add_constraint(Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=10.0
        ))  # version 3

        # Get changes since version 1
        changes = model.get_changes_since(1)
        assert len(changes) == 2
        assert changes[0].change_type == ChangeType.VARIABLE_ADDED
        assert changes[0].entity_name == "y"
        assert changes[1].change_type == ChangeType.CONSTRAINT_ADDED
        assert changes[1].entity_name == "c1"

    def test_get_changes_since_negative_version(self):
        """Should return all changes if version is negative"""
        model = Model(name="test")
        model.add_variable(Variable(name="x"))
        model.add_variable(Variable(name="y"))

        changes = model.get_changes_since(-1)
        assert len(changes) == 2
        
        changes = model.get_changes_since(-5)
        assert len(changes) == 2

    def test_clear_changes(self):
        """Should clear the change log"""
        model = Model(name="test")
        model.add_variable(Variable(name="x"))
        model.add_constraint(Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=10.0
        ))

        model.clear_changes()

        assert len(model._changes) == 0
        # Version should remain unchanged
        assert model.get_version() == 2

    def test_copy_clears_changes(self):
        """Copying a model should clear its change log"""
        model = Model(name="test")
        model.add_variable(Variable(name="x"))
        model.add_constraint(Constraint(
            name="c1",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=10.0
        ))

        model_copy = model.copy()

        # Copy should have cleared changes
        assert len(model_copy._changes) == 0
        # But should have same version
        assert model_copy.get_version() == model.get_version()
        # Original should still have changes
        assert len(model._changes) == 2

    def test_multiple_operations_sequence(self):
        """Test a complex sequence of operations"""
        model = Model(name="test")

        # Add variables
        model.add_variable(Variable(name="x"))
        model.add_variable(Variable(name="y"))

        # Add constraints
        model.add_constraint(Constraint(
            name="c1",
            coefficients={"x": 1.0, "y": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=10.0
        ))
        model.add_constraint(Constraint(
            name="c2",
            coefficients={"x": 2.0, "y": -1.0},
            constraint_type=ConstraintType.GEQ,
            rhs=5.0
        ))

        # Deactivate one
        model.deactivate_constraint("c1")

        # Reactivate it
        model.activate_constraint("c1")

        # Remove one
        model.remove_constraint("c2")

        assert model.get_version() == 7
        assert len(model._changes) == 7

    def test_remove_nonexistent_constraint_no_change(self):
        """Removing a constraint that doesn't exist should not log change"""
        model = Model(name="test")
        initial_version = model.get_version()

        model.remove_constraint("nonexistent")

        # No change should be logged
        assert model.get_version() == initial_version
        assert len(model._changes) == 0

    def test_deactivate_nonexistent_constraint_no_change(self):
        """Deactivating a constraint that doesn't exist should not log change"""
        model = Model(name="test")
        initial_version = model.get_version()

        model.deactivate_constraint("nonexistent")

        # No change should be logged
        assert model.get_version() == initial_version
        assert len(model._changes) == 0
