"""
Tests for incremental model updates in HiGHS solver
"""

import pytest
import time
from core.model import Model
from core.base import Variable, Constraint, Objective, VariableType, ConstraintType, SolutionStatus
from solvers.highs import HighsSolver, HighsStatus, HighsModelStatus, kHighsInf


class TestIncrementalUpdates:
    """Test incremental constraint operations"""

    def test_sync_model_initial_load(self):
        """First sync_model call should behave like load_model"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=5.0
        ))

        solver = HighsSolver()
        solver.sync_model(model)

        assert solver.model is not None
        assert solver.synced_model_version == model.get_version()
        assert "c1" in solver.constraint_indices

    def test_sync_model_no_changes(self):
        """sync_model with no changes should do nothing"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        solver = HighsSolver()
        solver.load_model(model)
        initial_version = solver.synced_model_version

        # Sync again with no changes
        solver.sync_model(model)

        assert solver.synced_model_version == initial_version

    def test_add_constraint_incremental(self):
        """Adding a constraint should use incremental update"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        assert abs(solution["x"] - 0.0) < 1e-6  # x should be 0 (minimizing)

        # Add constraint x >= 5
        solver.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=5.0
        ))

        # Solve again
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        assert abs(solution["x"] - 5.0) < 1e-6  # x should now be 5

        # Check constraint was added to index
        assert "c1" in solver.constraint_indices
        
    def test_add_constraint_incremental_via_updated_model_and_not_synced(self):
        """Adding a constraint should use incremental update"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        assert abs(solution["x"] - 0.0) < 1e-6  # x should be 0 (minimizing)

        # Add constraint x >= 5 to the model, not yet to the solver's model
        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=5.0
        ))

        # Solve again	# This should detect the un-synced changes and auto-sync the model
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        assert abs(solution["x"] - 5.0) < 1e-6  # x should now be 5

        # Check constraint was added to index
        assert "c1" in solver.constraint_indices

    def test_remove_constraint_incremental(self):
        """Removing a constraint should use incremental update"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=5.0
        ))

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        assert abs(solution["x"] - 5.0) < 1e-6  # x should be 5

        # Remove constraint
        solver.remove_constraint("c1")

        # Solve again
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        assert abs(solution["x"] - 0.0) < 1e-6  # x should now be 0

        # Check constraint was removed from index
        assert "c1" not in solver.constraint_indices

    def test_remove_constraint_incremental_with_2_constraints(self):
        """Removing a constraint should use incremental update"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=5.0
        ))
        model.add_constraint(Constraint(
            name="c2", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=7.0
        ))

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        assert abs(solution["x"] - 7.0) < 1e-6  # x should be 7

        # Remove constraint
        solver.remove_constraint("c2")

        # Solve again
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = solver.get_solution()
        assert abs(solution["x"] - 5.0) < 1e-6  # x should now be 5

        # Check constraint was removed from index
        assert "c2" not in solver.constraint_indices

    def test_deactivate_constraint_incremental(self):
        """Deactivating a constraint should use bounds relaxation"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=5.0
        ))

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()
        solution = solver.get_solution()
        assert abs(solution["x"] - 5.0) < 1e-6

		# Check bounds before deactivating
        status, row_id = solver.highs.getRowByName("c1")
        assert status == HighsStatus.kOk, "Failed to get row by name"
        status, lb, ub, nnz = solver.highs.getRow(row_id)
        assert status == HighsStatus.kOk, "Failed to get row"
        assert lb == 5.0 and ub == kHighsInf

        # Deactivate constraint
        solver.set_constraint_active("c1", False)
        status, lb, ub, nnz = solver.highs.getRow(row_id)
        assert status == HighsStatus.kOk, "Failed to get row after deactivating"
        assert lb == -kHighsInf and ub == kHighsInf

        # Solve again
        status = solver.solve()
        solution = solver.get_solution()
        assert abs(solution["x"] - 0.0) < 1e-6  # x should now be 0

        # Constraint should still be in index (just relaxed)
        assert "c1" in solver.constraint_indices

    def test_activate_constraint_incremental(self):
        """Reactivating a constraint should restore bounds"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=5.0
        ))

        solver = HighsSolver()
        solver.load_model(model)

        # Deactivate then reactivate
        # Check bounds before deactivating
        status, row_id = solver.highs.getRowByName("c1")
        assert status == HighsStatus.kOk, "Failed to get row by name"
        status, lb, ub, nnz = solver.highs.getRow(row_id)
        assert status == HighsStatus.kOk, "Failed to get row"
        assert lb == 5.0 and ub == kHighsInf
        solver.set_constraint_active("c1", False)
        status, lb, ub, nnz = solver.highs.getRow(row_id)
        assert status == HighsStatus.kOk, "Failed to get row after deactivating"
        assert lb == -kHighsInf and ub == kHighsInf
        solver.set_constraint_active("c1", True)
        status, lb, ub, nnz = solver.highs.getRow(row_id)
        assert status == HighsStatus.kOk, "Failed to get row after reactivating"
        assert lb == 5.0 and ub == kHighsInf

        # Solve
        status = solver.solve()
        solution = solver.get_solution()
        assert abs(solution["x"] - 5.0) < 1e-6  # x should be 5 again

    def test_multiple_constraint_toggles(self):
        """Test toggling constraints multiple times"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=3.0
        ))
        model.add_constraint(Constraint(
            name="c2", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=7.0
        ))

        solver = HighsSolver()
        solver.load_model(model)

        # With both constraints active, x should be 7
        status = solver.solve()
        solution = solver.get_solution()
        assert abs(solution["x"] - 7.0) < 1e-6

        # Deactivate c2, x should be 3
        solver.set_constraint_active("c2", False)
        status = solver.solve()
        solution = solver.get_solution()
        assert abs(solution["x"] - 3.0) < 1e-6

        # Deactivate c1, x should be 0
        solver.set_constraint_active("c1", False)
        status = solver.solve()
        solution = solver.get_solution()
        assert abs(solution["x"] - 0.0) < 1e-6

        # Reactivate c1, x should be 3
        solver.set_constraint_active("c1", True)
        status = solver.solve()
        solution = solver.get_solution()
        assert abs(solution["x"] - 3.0) < 1e-6

        # Reactivate c2, x should be 7
        solver.set_constraint_active("c2", True)
        status = solver.solve()
        solution = solver.get_solution()
        assert abs(solution["x"] - 7.0) < 1e-6

    def test_remove_then_add_constraint(self):
        """Test removing and re-adding a constraint"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        solver = HighsSolver()
        solver.load_model(model)

        # Add constraint
        solver.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=5.0
        ))
        status = solver.solve()
        solution = solver.get_solution()
        assert abs(solution["x"] - 5.0) < 1e-6

        # Remove constraint
        solver.remove_constraint("c1")
        status = solver.solve()
        solution = solver.get_solution()
        assert abs(solution["x"] - 0.0) < 1e-6

        # Add it back
        solver.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=5.0
        ))
        status = solver.solve()
        solution = solver.get_solution()
        assert abs(solution["x"] - 5.0) < 1e-6

    def test_index_mapping_after_remove(self):
        """Test that indices update correctly after removing a constraint"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
        model.add_constraint(Constraint(name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=1.0))
        model.add_constraint(Constraint(name="c2", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=2.0))
        model.add_constraint(Constraint(name="c3", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=3.0))

        solver = HighsSolver()
        solver.load_model(model)

        # Initial indices: c1=0, c2=1, c3=2
        assert solver.constraint_indices["c1"] == 0
        assert solver.constraint_indices["c2"] == 1
        assert solver.constraint_indices["c3"] == 2

        # Remove c2 (middle)
        solver.remove_constraint("c2")

        # Indices should shift: c1=0, c3=1
        assert solver.constraint_indices["c1"] == 0
        assert "c2" not in solver.constraint_indices
        assert solver.constraint_indices["c3"] == 1

    def test_mixed_operations_sequence(self):
        """Test a complex sequence of add, remove, toggle operations"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=20.0)
        y = Variable(name="y", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=20.0)
        model.add_variable(x)
        model.add_variable(y)
        model.objective = Objective(coefficients={"x": 1.0, "y": 1.0}, is_minimize=True)

        solver = HighsSolver()
        solver.load_model(model)

        # Add constraints incrementally
        solver.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0, "y": 1.0}, constraint_type=ConstraintType.GEQ, rhs=10.0
        ))
        solver.add_constraint(Constraint(
            name="c2", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=3.0
        ))
        solver.add_constraint(Constraint(
            name="c3", coefficients={"y": 1.0}, constraint_type=ConstraintType.GEQ, rhs=4.0
        ))

        # Solve - should get x=3, y=7 or x=6, y=4, etc. (x+y=10, x>=3, y>=4)
        status = solver.solve()
        solution = solver.get_solution()
        assert abs(solution["x"] + solution["y"] - 10.0) < 1e-6

        # Deactivate c1
        solver.set_constraint_active("c1", False)
        status = solver.solve()
        solution = solver.get_solution()
        # Should minimize to x=3, y=4
        assert abs(solution["x"] - 3.0) < 1e-6
        assert abs(solution["y"] - 4.0) < 1e-6

        # Remove c2
        solver.remove_constraint("c2")
        status = solver.solve()
        solution = solver.get_solution()
        # Should minimize to x=0, y=4
        assert abs(solution["x"] - 0.0) < 1e-6
        assert abs(solution["y"] - 4.0) < 1e-6

    def test_infeasibility_detection_after_constraint_add(self):
        """Test that adding a conflicting constraint is detected as infeasible"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.LEQ, rhs=5.0
        ))

        solver = HighsSolver() 
        solver.load_model(model)
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL

        # Add conflicting constraint
        solver.add_constraint(Constraint(
            name="c2", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=10.0
        ))

        status = solver.solve()
        assert status == SolutionStatus.INFEASIBLE

    def test_feasibility_restored_after_constraint_remove(self):
        """Test that removing a conflicting constraint restores feasibility"""
        model = Model(name="test")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.LEQ, rhs=5.0
        ))
        model.add_constraint(Constraint(
            name="c2", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=10.0
        ))

        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()
        assert status == SolutionStatus.INFEASIBLE

        # Remove conflicting constraint
        solver.remove_constraint("c2")

        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
