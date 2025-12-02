"""
Base classes and interfaces for IIS Finder
Author: Surya Krishnan
Date: 2025-11-20
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Set, Dict, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from core.model import Model
    from core.solver_interface import SolverInterface


class ConstraintType(Enum):
    """Types of constraints in optimization models"""
    LEQ = "<="  # Less than or equal
    EQ = "="    # Equal
    GEQ = ">="  # Greater than or equal


class VariableType(Enum):
    """Types of variables in optimization models"""
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


class SolutionStatus(Enum):
    """Solution status from solver"""
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    UNKNOWN = "unknown"
    TIME_LIMIT = "time_limit"
    ERROR = "error"


class ChangeType(Enum):
    """Types of changes that can occur to a model"""
    CONSTRAINT_ADDED = "constraint_added"
    CONSTRAINT_REMOVED = "constraint_removed"
    CONSTRAINT_ACTIVATED = "constraint_activated"
    CONSTRAINT_DEACTIVATED = "constraint_deactivated"
    VARIABLE_ADDED = "variable_added"
    OBJECTIVE_CHANGED = "objective_changed"


@dataclass
class Variable:
    """Represents a decision variable"""
    name: str
    var_type: VariableType = VariableType.CONTINUOUS
    lower_bound: float = 0.0
    upper_bound: float = float('inf')
    index: Optional[int] = None
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class Constraint:
    """Represents a linear constraint"""
    name: str
    coefficients: Dict[str, float]  # variable_name -> coefficient
    constraint_type: ConstraintType
    rhs: float
    index: Optional[int] = None
    is_active: bool = True
    
    def __hash__(self):
        return hash(self.name)
    
    def __str__(self):
        terms = []
        for var, coef in self.coefficients.items():
            if coef != 0:
                if coef == 1:
                    terms.append(var)
                elif coef == -1:
                    terms.append(f"-{var}")
                else:
                    terms.append(f"{coef}*{var}")
        
        lhs = " + ".join(terms) if terms else "0"
        return f"{self.name}: {lhs} {self.constraint_type.value} {self.rhs}"


@dataclass
class Objective:
    """Represents the objective function"""
    coefficients: Dict[str, float]  # variable_name -> coefficient
    is_minimize: bool = True
    constant: float = 0.0


@dataclass
class ModelChange:
    """Represents a single change to the model"""
    change_type: ChangeType
    entity_name: str  # Name of variable/constraint affected
    entity: Optional[Any] = None  # The actual object (for adds)


@dataclass
class IISResult:
    """Result from IIS finding algorithm"""
    iis_constraints: Set[str]  # Names of constraints in the IIS
    algorithm: str
    time_elapsed: float
    iterations: int = 0
    status: str = "success"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return (f"IIS Result:\n"
                f"  Algorithm: {self.algorithm}\n"
                f"  Status: {self.status}\n"
                f"  IIS Size: {len(self.iis_constraints)}\n"
                f"  Time: {self.time_elapsed:.3f}s\n"
                f"  Iterations: {self.iterations}\n"
                f"  Constraints in IIS: {sorted(self.iis_constraints)}")


class IISAlgorithm(ABC):
    """Abstract base class for IIS finding algorithms"""

    def __init__(self, solver: 'SolverInterface'):
        self.solver = solver
        self.model: Optional['Model'] = None

    @abstractmethod
    def find_iis(self, model: 'Model') -> IISResult:
        """Find an IIS in the given model"""
        pass

    def verify_iis(self, model: 'Model', iis_constraints: Set[str]) -> bool:
        """
        Verify that a set of constraints forms an IIS

        Returns:
            True if the constraints form a valid IIS
        """
        from core.model import Model
        # Create a model with only IIS constraints
        iis_model = Model(name=f"{model.name}_iis_verification")
        iis_model.variables = model.variables.copy()
        iis_model.objective = model.objective
        
        for constraint_name in iis_constraints:
            if constraint_name in model.constraints:
                iis_model.add_constraint(model.constraints[constraint_name])
        
        # Check that IIS is infeasible
        self.solver.load_model(iis_model)
        status = self.solver.solve()
        if status != SolutionStatus.INFEASIBLE:
            return False
        
        # Check that removing any constraint makes it feasible
        for constraint_name in iis_constraints:
            test_model = iis_model.copy()
            test_model.remove_constraint(constraint_name)
            self.solver.load_model(test_model)
            status = self.solver.solve()
            if status == SolutionStatus.INFEASIBLE:
                return False  # Not minimal
        
        return True