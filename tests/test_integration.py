"""
Integration tests and performance benchmarks for incremental updates
"""

import pytest
import time
from core.model import Model
from core.base import Variable, Constraint, Objective, VariableType, ConstraintType, SolutionStatus
from solvers.highs import HighsSolver, HighsStatus, HighsModelStatus, kHighsInf


class TestIntegration:
    """Integration tests combining multiple features"""

    def test_full_workflow(self):
        """Test a complete workflow: build model, solve, modify, resolve"""
        # Build initial model
        model = Model(name="workflow_test")

        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=100.0)
        y = Variable(name="y", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=100.0)
        model.add_variable(x)
        model.add_variable(y)

        model.objective = Objective(coefficients={"x": 2.0, "y": 3.0}, is_minimize=True)

        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0, "y": 1.0}, constraint_type=ConstraintType.GEQ, rhs=20.0
        ))
        model.add_constraint(Constraint(
            name="c2", coefficients={"x": 2.0, "y": 1.0}, constraint_type=ConstraintType.GEQ, rhs=30.0
        ))

        # Load and solve
        solver = HighsSolver()
        solver.load_model(model)
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
        initial_obj = solver.get_objective_value()

		# choose a constraint whose dual is positive
        duals = solver.get_dual_values()
        is_pos_dual = [d > 1e-6 for d in duals.values()]
        assert len(is_pos_dual) > 0, "No positive dual values found, this test definition needs to have at least one constraint with positive dual value"
        row_names_pos = [row_name for row_name, val in zip(duals.keys(), is_pos_dual) if val]
        import random
        row_name = random.choice(row_names_pos)
        
		# deactive row with positive dual
        solver.set_constraint_active(row_name, False)
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
        modified_obj = solver.get_objective_value()

        # # Modify model: deactivate c1
        # solver.set_constraint_active("c1", False)
        # status = solver.solve()
        # assert status == SolutionStatus.OPTIMAL
        # modified_obj = solver.get_objective_value()

        # Objective should improve (decrease) when we relax a binding constraint
        assert modified_obj < initial_obj

        # Reactivate c2
        solver.set_constraint_active(row_name, True)
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
        restored_obj = solver.get_objective_value()

        # Should get back to original objective
        assert abs(restored_obj - initial_obj) < 1e-6

    def test_model_copy_with_solver(self):
        """Test that model copying works correctly with solver"""
        model = Model(name="original")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0, upper_bound=10.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=5.0
        ))

        # Solve original
        solver1 = HighsSolver()
        solver1.load_model(model)
        status = solver1.solve()
        assert status == SolutionStatus.OPTIMAL
        sol1 = solver1.get_solution()

        # Copy model and modify
        model_copy = model.copy()
        model_copy.deactivate_constraint("c1")

        # Solve copy
        solver2 = HighsSolver()
        solver2.load_model(model_copy)
        status = solver2.solve()
        assert status == SolutionStatus.OPTIMAL
        sol2 = solver2.get_solution()

        # Solutions should be different
        assert abs(sol1["x"] - 5.0) < 1e-6
        assert abs(sol2["x"] - 0.0) < 1e-6

        # Original solver should still work correctly
        status = solver1.solve()
        sol1_again = solver1.get_solution()
        assert abs(sol1_again["x"] - 5.0) < 1e-6

    def test_constraint_activation_pattern(self):
        """Test a pattern similar to IIS deletion filter algorithm"""
        # This mimics how deletion filter would work:
        # 1. Start with all constraints
        # 2. Try removing each constraint one by one
        # 3. Check if still infeasible

        model = Model(name="iis_pattern")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)

        # Create an infeasible set: x <= 5, x >= 10
        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.LEQ, rhs=5.0
        ))
        model.add_constraint(Constraint(
            name="c2", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=10.0
        ))

        solver = HighsSolver()
        solver.load_model(model)

        # Should be infeasible with both constraints
        status = solver.solve()
        assert status == SolutionStatus.INFEASIBLE

        # Try removing c1
        solver.set_constraint_active("c1", False)
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL  # Feasible without c1

        # Restore c1
        solver.set_constraint_active("c1", True)
        status = solver.solve()
        assert status == SolutionStatus.INFEASIBLE

        # Try removing c2
        solver.set_constraint_active("c2", False)
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL  # Feasible without c2

        # Both constraints together form an IIS


class TestPerformance:
    """Performance benchmarks for incremental updates"""

    def test_benchmark_constraint_toggles(self):
        """Benchmark toggling constraints with incremental updates"""
        # Create a model with multiple constraints
        model = Model(name="benchmark")

        num_vars = 10
        for i in range(num_vars):
            model.add_variable(Variable(
                name=f"x{i}",
                var_type=VariableType.CONTINUOUS,
                lower_bound=0.0,
                upper_bound=100.0
            ))

        # Minimize sum of all variables
        model.objective = Objective(
            coefficients={f"x{i}": 1.0 for i in range(num_vars)},
            is_minimize=True
        )

        # Add 20 constraints
        num_constraints = 20
        for i in range(num_constraints):
            # x_i >= i
            model.add_constraint(Constraint(
                name=f"c{i}",
                coefficients={f"x{i % num_vars}": 1.0},
                constraint_type=ConstraintType.GEQ,
                rhs=float(i)
            ))

        solver = HighsSolver()
        solver.load_model(model)

        # Benchmark: toggle each constraint on/off 10 times
        num_toggles = 10
        constraint_names = [f"c{i}" for i in range(num_constraints)]

        start_time = time.time()

        for toggle_round in range(num_toggles):
            for c_name in constraint_names:
                # Deactivate
                solver.set_constraint_active(c_name, False)
                solver.solve()

                # Reactivate
                solver.set_constraint_active(c_name, True)
                solver.solve()

        elapsed = time.time() - start_time

        total_operations = num_toggles * num_constraints * 2 * 2  # toggles * constraints * (on+off) * (toggle+solve)
        avg_time_per_op = elapsed / total_operations

        print(f"\nPerformance benchmark:")
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Total operations: {total_operations}")
        print(f"  Average time per operation: {avg_time_per_op*1000:.2f}ms")

        # Assert reasonable performance (< 50ms per operation)
        assert avg_time_per_op < 0.05, f"Operations too slow: {avg_time_per_op*1000:.2f}ms per operation"

    def test_comparison_incremental_vs_full_reload(self):
        """Compare performance of incremental updates vs full reload"""
        # Create a model
        model = Model(name="comparison")

        for i in range(5):
            model.add_variable(Variable(
                name=f"x{i}",
                var_type=VariableType.CONTINUOUS,
                lower_bound=0.0,
                upper_bound=10.0
            ))

        model.objective = Objective(
            coefficients={f"x{i}": 1.0 for i in range(5)},
            is_minimize=True
        )

        for i in range(10):
            model.add_constraint(Constraint(
                name=f"c{i}",
                coefficients={f"x{i % 5}": 1.0},
                constraint_type=ConstraintType.GEQ,
                rhs=float(i % 5)
            ))

        # Test incremental updates
        solver_incremental = HighsSolver()
        solver_incremental.load_model(model)

        start_incremental = time.time()
        for _ in range(50):
            solver_incremental.set_constraint_active("c5", False)
            solver_incremental.solve()
            solver_incremental.set_constraint_active("c5", True)
            solver_incremental.solve()
        elapsed_incremental = time.time() - start_incremental

        # Test full reload (simulated)
        # Note: We can't easily test full reload directly, but we can measure load_model time
        start_reload = time.time()
        for _ in range(50):
            solver_reload = HighsSolver()
            solver_reload.load_model(model)
            solver_reload.solve()
        elapsed_reload = time.time() - start_reload

        print(f"\nComparison:")
        print(f"  Incremental updates: {elapsed_incremental:.3f}s")
        print(f"  Full reload approach: {elapsed_reload:.3f}s")
        print(f"  Speedup: {elapsed_reload/elapsed_incremental:.1f}x")

        # Incremental should be significantly faster
        assert elapsed_incremental < elapsed_reload, "Incremental updates should be faster than full reload"

    def test_large_constraint_set_handling(self):
        """Test that incremental updates work correctly with many constraints"""
        model = Model(name="large")

        # Create variables
        for i in range(10):
            model.add_variable(Variable(
                name=f"x{i}",
                var_type=VariableType.CONTINUOUS,
                lower_bound=0.0,
                upper_bound=100.0
            ))

        model.objective = Objective(
            coefficients={f"x{i}": float(i+1) for i in range(10)},
            is_minimize=True
        )

        # Add 50 constraints
        for i in range(50):
            model.add_constraint(Constraint(
                name=f"c{i}",
                coefficients={f"x{i % 10}": 1.0, f"x{(i+1) % 10}": 1.0},
                constraint_type=ConstraintType.GEQ,
                rhs=float(i % 10)
            ))

        solver = HighsSolver()
        solver.load_model(model)

        # Should solve successfully
        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
        original_objective_value = solver.get_objective_value()

        # Deactivate half the constraints
        for i in range(0, 50, 2):
            solver.set_constraint_active(f"c{i}", False)

        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
        modified_objective_value = solver.get_objective_value()

        # Reactivate them
        for i in range(0, 50, 2):
            solver.set_constraint_active(f"c{i}", True)

        status = solver.solve()
        assert status == SolutionStatus.OPTIMAL
        reactivated_objective_value = solver.get_objective_value()
        assert original_objective_value == reactivated_objective_value
        
        # Check all constraints are properly tracked
        assert len(solver.constraint_indices) == 50

    def test_memory_efficiency(self):
        """Test that change log doesn't grow unbounded"""
        model = Model(name="memory")
        x = Variable(name="x", var_type=VariableType.CONTINUOUS, lower_bound=0.0)
        model.add_variable(x)
        model.objective = Objective(coefficients={"x": 1.0}, is_minimize=True)
        model.add_constraint(Constraint(
            name="c1", coefficients={"x": 1.0}, constraint_type=ConstraintType.GEQ, rhs=5.0
        ))

        solver = HighsSolver()
        solver.load_model(model)

        # Make many changes
        for i in range(100):
            solver.set_constraint_active("c1", False)
            solver.set_constraint_active("c1", True)

        # Model change log should be manageable
        # (In practice, solver.sync_model could clear changes after syncing,
        # but currently we don't do that automatically)
        # This test just verifies the system doesn't crash with many changes
        assert model.get_version() == 202  # 2 (initial) + 2*100 (toggles)
