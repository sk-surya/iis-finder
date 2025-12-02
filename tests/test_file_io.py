"""
Tests for file I/O functionality
"""

import pytest
import tempfile
import os
from core.base import VariableType, ConstraintType, SolutionStatus
from solvers.highs import HighsSolver


class TestFileIO:
    """Test loading models from files"""

    def test_load_from_lp_file(self):
        """Test loading a model from LP format file"""
        # Create a simple LP file
        lp_content = """
Minimize
 obj: x + 2 y
Subject To
 c1: x + y >= 5
 c2: 2 x + y >= 8
Bounds
 0 <= x <= 10
 0 <= y <= 10
End
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as f:
            f.write(lp_content)
            filename = f.name

        try:
            solver = HighsSolver()
            model = solver.load_from_file(filename)

            # Check model was loaded
            assert model is not None
            assert model.name is not None

            # Check variables were created
            assert len(model.variables) > 0

            # Check constraints were created
            assert len(model.constraints) > 0

            # Try solving the loaded model
            solver.load_model(model)
            status = solver.solve()
            assert status == SolutionStatus.OPTIMAL

        finally:
            os.unlink(filename)

    def test_load_from_mps_file(self):
        """Test loading a model from MPS format file"""
        # Create a simple MPS file
        mps_content = """NAME          TESTPROB
ROWS
 N  OBJ
 G  C1
 G  C2
COLUMNS
    X1        OBJ       1.0
    X1        C1        1.0
    X1        C2        2.0
    X2        OBJ       2.0
    X2        C1        1.0
    X2        C2        1.0
RHS
    RHS1      C1        5.0
    RHS1      C2        8.0
BOUNDS
 UP BND1      X1        10.0
 UP BND1      X2        10.0
ENDATA
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mps', delete=False) as f:
            f.write(mps_content)
            filename = f.name

        try:
            solver = HighsSolver()
            model = solver.load_from_file(filename)

            # Check model was loaded
            assert model is not None

            # Check variables were created
            assert len(model.variables) >= 2

            # Check constraints were created
            assert len(model.constraints) >= 2

            # Try solving the loaded model
            solver.load_model(model)
            status = solver.solve()
            assert status in [SolutionStatus.OPTIMAL, SolutionStatus.INFEASIBLE]

        finally:
            os.unlink(filename)

    def test_load_infeasible_model_from_file(self):
        """Test loading an infeasible model from file"""
        # Create an infeasible LP file: x <= 5 and x >= 10
        lp_content = """
Minimize
 obj: x
Subject To
 c1: x <= 5
 c2: x >= 10
End
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as f:
            f.write(lp_content)
            filename = f.name

        try:
            solver = HighsSolver()
            model = solver.load_from_file(filename)

            # Load and solve
            solver.load_model(model)
            status = solver.solve()

            # Should be infeasible
            assert status == SolutionStatus.INFEASIBLE

        finally:
            os.unlink(filename)

    def test_load_binary_variables_from_file(self):
        """Test loading a model with binary variables"""
        lp_content = """
Maximize
 obj: x + 2 y
Subject To
 c1: x + y <= 1
Binary
 x
 y
End
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as f:
            f.write(lp_content)
            filename = f.name

        try:
            solver = HighsSolver()
            model = solver.load_from_file(filename)

            # Check that variables were identified (though may not have exact names)
            assert len(model.variables) >= 2

            # Solve the model
            solver.load_model(model)
            status = solver.solve()
            assert status == SolutionStatus.OPTIMAL

            solution = solver.get_solution()
            # All values should be 0 or 1 (binary)
            for val in solution.values():
                assert val in [0.0, 1.0] or abs(val - round(val)) < 1e-6

        finally:
            os.unlink(filename)

    def test_load_integer_variables_from_file(self):
        """Test loading a model with integer variables"""
        lp_content = """
Minimize
 obj: x + y
Subject To
 c1: 2 x + 3 y >= 10
Bounds
 0 <= x <= 10
 0 <= y <= 10
General
 x
 y
End
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as f:
            f.write(lp_content)
            filename = f.name

        try:
            solver = HighsSolver()
            model = solver.load_from_file(filename)

            # Check variables were loaded
            assert len(model.variables) >= 2

            # Solve the model
            solver.load_model(model)
            status = solver.solve()
            assert status == SolutionStatus.OPTIMAL

            solution = solver.get_solution()
            # All values should be integers
            for val in solution.values():
                assert abs(val - round(val)) < 1e-6

        finally:
            os.unlink(filename)
