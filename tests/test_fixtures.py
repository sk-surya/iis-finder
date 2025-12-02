"""
Tests that use fixtures from conftest.py to increase coverage
"""

import pytest
from core.base import SolutionStatus, ConstraintType, Constraint
from tests.conftest import assert_solution_close, assert_objective_close


class TestSimpleModelFixture:
    """Tests using simple_model fixture"""

    def test_simple_model_structure(self, simple_model):
        """Test simple_model fixture creates correct structure"""
        assert simple_model.name == "simple"
        assert len(simple_model.variables) == 1
        assert "x" in simple_model.variables
        assert simple_model.variables["x"].lower_bound == 0.0
        assert simple_model.variables["x"].upper_bound == 10.0
        assert simple_model.objective is not None
        assert simple_model.objective.is_minimize is True

    def test_simple_model_solve(self, simple_model, highs_solver):
        """Test solving simple_model fixture"""
        highs_solver.load_model(simple_model)
        status = highs_solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = highs_solver.get_solution()
        # Should minimize to x=0
        assert abs(solution["x"] - 0.0) < 1e-6

    def test_simple_model_with_constraint(self, simple_model, highs_solver):
        """Test simple_model with added constraint"""
        simple_model.add_constraint(Constraint(
            name="lower_bound",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.GEQ,
            rhs=5.0
        ))
        highs_solver.load_model(simple_model)
        status = highs_solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = highs_solver.get_solution()
        # Should minimize to x=5
        assert abs(solution["x"] - 5.0) < 1e-6


class TestTwoVarModelFixture:
    """Tests using two_var_model fixture"""

    def test_two_var_model_structure(self, two_var_model):
        """Test two_var_model fixture creates correct structure"""
        assert two_var_model.name == "two_var"
        assert len(two_var_model.variables) == 2
        assert "x" in two_var_model.variables
        assert "y" in two_var_model.variables
        assert two_var_model.objective is not None

    def test_two_var_model_solve(self, two_var_model, highs_solver):
        """Test solving two_var_model fixture"""
        highs_solver.load_model(two_var_model)
        status = highs_solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = highs_solver.get_solution()
        # Should minimize to x=0, y=0
        assert abs(solution["x"] - 0.0) < 1e-6
        assert abs(solution["y"] - 0.0) < 1e-6

    def test_two_var_model_with_constraint(self, two_var_model, highs_solver):
        """Test two_var_model with constraint"""
        two_var_model.add_constraint(Constraint(
            name="sum_constraint",
            coefficients={"x": 1.0, "y": 1.0},
            constraint_type=ConstraintType.GEQ,
            rhs=8.0
        ))
        highs_solver.load_model(two_var_model)
        status = highs_solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = highs_solver.get_solution()
        # Should have x + y = 8 (with some combination)
        assert abs(solution["x"] + solution["y"] - 8.0) < 1e-6


class TestFeasibleModelFixture:
    """Tests using feasible_model fixture"""

    def test_feasible_model_structure(self, feasible_model):
        """Test feasible_model fixture structure"""
        assert feasible_model.name == "feasible"
        assert len(feasible_model.variables) == 2
        assert len(feasible_model.constraints) == 1
        assert "c1" in feasible_model.constraints

    def test_feasible_model_solve(self, feasible_model, highs_solver):
        """Test solving feasible_model fixture"""
        highs_solver.load_model(feasible_model)
        status = highs_solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = highs_solver.get_solution()
        # x + y should be at least 10, and we minimize x + y, so it should be 10
        assert abs(solution["x"] + solution["y"] - 10.0) < 1e-6

    def test_feasible_model_optimal_value(self, feasible_model, highs_solver):
        """Test feasible_model objective value"""
        highs_solver.load_model(feasible_model)
        highs_solver.solve()
        obj_value = highs_solver.get_objective_value()
        # Minimizing x + y with x + y >= 10 gives objective = 10
        assert abs(obj_value - 10.0) < 1e-6


class TestInfeasibleModelFixture:
    """Tests using infeasible_model fixture"""

    def test_infeasible_model_structure(self, infeasible_model):
        """Test infeasible_model fixture structure"""
        assert infeasible_model.name == "infeasible"
        assert len(infeasible_model.variables) == 1
        assert len(infeasible_model.constraints) == 2
        assert "c1" in infeasible_model.constraints
        assert "c2" in infeasible_model.constraints

    def test_infeasible_model_solve(self, infeasible_model, highs_solver):
        """Test that infeasible_model is detected as infeasible"""
        highs_solver.load_model(infeasible_model)
        status = highs_solver.solve()
        assert status == SolutionStatus.INFEASIBLE

    def test_infeasible_model_becomes_feasible_removing_c1(self, infeasible_model, highs_solver):
        """Test that removing c1 makes model feasible"""
        infeasible_model.remove_constraint("c1")
        highs_solver.load_model(infeasible_model)
        status = highs_solver.solve()
        # Only c2 remains (x >= 10), which is feasible
        assert status == SolutionStatus.OPTIMAL

    def test_infeasible_model_becomes_feasible_removing_c2(self, infeasible_model, highs_solver):
        """Test that removing c2 makes model feasible"""
        infeasible_model.remove_constraint("c2")
        highs_solver.load_model(infeasible_model)
        status = highs_solver.solve()
        # Only c1 remains (x <= 5), which is feasible
        assert status == SolutionStatus.OPTIMAL


class TestLoadedSolverFixture:
    """Tests using loaded_solver fixture"""

    def test_loaded_solver_already_has_model(self, loaded_solver):
        """Test that loaded_solver fixture has a model loaded"""
        assert loaded_solver.model is not None
        assert loaded_solver.synced_model_version >= 0

    def test_loaded_solver_can_solve(self, loaded_solver):
        """Test that loaded_solver can solve immediately"""
        status = loaded_solver.solve()
        assert status == SolutionStatus.OPTIMAL

    def test_loaded_solver_has_solution(self, loaded_solver):
        """Test that loaded_solver can get solution"""
        loaded_solver.solve()
        solution = loaded_solver.get_solution()
        assert solution is not None
        assert "x" in solution
        assert "y" in solution


class TestMultiConstraintModelFixture:
    """Tests using multi_constraint_model fixture"""

    def test_multi_constraint_model_structure(self, multi_constraint_model):
        """Test multi_constraint_model fixture structure"""
        assert multi_constraint_model.name == "multi"
        assert len(multi_constraint_model.variables) == 1
        assert len(multi_constraint_model.constraints) == 3
        assert "c1" in multi_constraint_model.constraints
        assert "c2" in multi_constraint_model.constraints
        assert "c3" in multi_constraint_model.constraints

    def test_multi_constraint_model_solve(self, multi_constraint_model, highs_solver):
        """Test solving multi_constraint_model"""
        highs_solver.load_model(multi_constraint_model)
        status = highs_solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = highs_solver.get_solution()
        # The binding constraint is c3 (x >= 30), so x should be 30
        assert abs(solution["x"] - 30.0) < 1e-6

    def test_multi_constraint_model_toggle_constraints(self, multi_constraint_model, highs_solver):
        """Test toggling constraints in multi_constraint_model"""
        highs_solver.load_model(multi_constraint_model)

        # Deactivate c3 (the tightest constraint)
        highs_solver.set_constraint_active("c3", False)
        highs_solver.solve()
        solution = highs_solver.get_solution()
        # Now c2 should be binding (x >= 20)
        assert abs(solution["x"] - 20.0) < 1e-6

        # Deactivate c2 as well
        highs_solver.set_constraint_active("c2", False)
        highs_solver.solve()
        solution = highs_solver.get_solution()
        # Now only c1 is active (x >= 10)
        assert abs(solution["x"] - 10.0) < 1e-6


class TestStandardLPModelFixture:
    """Tests using standard_lp_model fixture"""

    def test_standard_lp_model_structure(self, standard_lp_model):
        """Test standard_lp_model fixture structure"""
        assert standard_lp_model.name == "standard_lp"
        assert len(standard_lp_model.variables) == 2
        assert len(standard_lp_model.constraints) == 3
        assert standard_lp_model.objective.is_minimize is False  # Maximization

    def test_standard_lp_model_solve(self, standard_lp_model, highs_solver):
        """Test solving standard_lp_model"""
        highs_solver.load_model(standard_lp_model)
        status = highs_solver.solve()
        assert status == SolutionStatus.OPTIMAL
        solution = highs_solver.get_solution()
        # Optimal solution: x=3, y=12
        assert abs(solution["x"] - 3.0) < 1e-6
        assert abs(solution["y"] - 12.0) < 1e-6

    def test_standard_lp_model_objective(self, standard_lp_model, highs_solver):
        """Test standard_lp_model objective value"""
        highs_solver.load_model(standard_lp_model)
        highs_solver.solve()
        obj_value = highs_solver.get_objective_value()
        # Optimal objective: 3*3 + 2*12 = 33
        assert abs(obj_value - 33.0) < 1e-6

    def test_standard_lp_model_constraints_satisfied(self, standard_lp_model, highs_solver):
        """Test that standard_lp_model solution satisfies all constraints"""
        highs_solver.load_model(standard_lp_model)
        highs_solver.solve()
        solution = highs_solver.get_solution()

        x = solution["x"]
        y = solution["y"]

        # Check all constraints are satisfied
        # c1: 2x + y <= 18
        assert 2*x + y <= 18.0 + 1e-6
        # c2: 2x + 3y <= 42
        assert 2*x + 3*y <= 42.0 + 1e-6
        # c3: 3x + y <= 24
        assert 3*x + y <= 24.0 + 1e-6


class TestHelperFunctions:
    """Tests using helper functions from conftest.py"""

    def test_assert_solution_close(self, feasible_model, highs_solver):
        """Test assert_solution_close helper"""
        highs_solver.load_model(feasible_model)
        highs_solver.solve()
        solution = highs_solver.get_solution()

        # This should pass - x + y = 10 (various combinations)
        # We know the sum should be 10, so let's check
        expected_sum = 10.0
        actual_sum = solution["x"] + solution["y"]
        assert abs(actual_sum - expected_sum) < 1e-6

        # Test the helper function works
        # Can't predict exact x and y, but we can test with actual values
        expected = {"x": solution["x"], "y": solution["y"]}
        assert_solution_close(solution, expected)

    def test_assert_objective_close(self, feasible_model, highs_solver):
        """Test assert_objective_close helper"""
        highs_solver.load_model(feasible_model)
        highs_solver.solve()

        # Objective should be 10.0
        assert_objective_close(highs_solver, 10.0)

    def test_assert_objective_close_with_standard_lp(self, standard_lp_model, highs_solver):
        """Test assert_objective_close with standard LP"""
        highs_solver.load_model(standard_lp_model)
        highs_solver.solve()

        # Objective should be 33.0 (3*3 + 2*12)
        assert_objective_close(highs_solver, 33.0)

    def test_assert_solution_close_with_standard_lp(self, standard_lp_model, highs_solver):
        """Test assert_solution_close with standard LP"""
        highs_solver.load_model(standard_lp_model)
        highs_solver.solve()
        solution = highs_solver.get_solution()

        # Expected solution: x=3, y=12
        expected = {"x": 3.0, "y": 12.0}
        assert_solution_close(solution, expected)


class TestFixtureCombinations:
    """Tests combining multiple fixtures"""

    def test_two_models_same_solver(self, simple_model, two_var_model, highs_solver):
        """Test loading different models into same solver"""
        # Load and solve first model
        highs_solver.load_model(simple_model)
        status1 = highs_solver.solve()
        assert status1 == SolutionStatus.OPTIMAL

        # Reset and load second model
        highs_solver.reset()
        highs_solver.load_model(two_var_model)
        status2 = highs_solver.solve()
        assert status2 == SolutionStatus.OPTIMAL

    def test_feasible_vs_infeasible(self, feasible_model, infeasible_model, highs_solver):
        """Test difference between feasible and infeasible models"""
        # Solve feasible model
        highs_solver.load_model(feasible_model)
        status1 = highs_solver.solve()
        assert status1 == SolutionStatus.OPTIMAL

        # Solve infeasible model
        highs_solver.reset()
        highs_solver.load_model(infeasible_model)
        status2 = highs_solver.solve()
        assert status2 == SolutionStatus.INFEASIBLE

    def test_model_modification_with_fixture(self, multi_constraint_model, highs_solver):
        """Test modifying a fixture model"""
        # Get original constraint count
        original_count = len(multi_constraint_model.constraints)

        # Add a new constraint
        multi_constraint_model.add_constraint(Constraint(
            name="c4",
            coefficients={"x": 1.0},
            constraint_type=ConstraintType.LEQ,
            rhs=50.0
        ))

        assert len(multi_constraint_model.constraints) == original_count + 1

        # Solve with new constraint
        highs_solver.load_model(multi_constraint_model)
        status = highs_solver.solve()
        assert status == SolutionStatus.OPTIMAL
