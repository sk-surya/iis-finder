"""
Base classes and interfaces for IIS Finder
Author: Surya Krishnan
Date: 2025-11-20
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Set, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field


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
class Model:
    """Represents an optimization model"""
    name: str = "model"
    variables: Dict[str, Variable] = field(default_factory=dict)
    constraints: Dict[str, Constraint] = field(default_factory=dict)
    objective: Optional[Objective] = None
    
    def add_variable(self, variable: Variable):
        """Add a variable to the model"""
        self.variables[variable.name] = variable
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the model"""
        self.constraints[constraint.name] = constraint
    
    def remove_constraint(self, constraint_name: str):
        """Remove a constraint from the model"""
        if constraint_name in self.constraints:
            del self.constraints[constraint_name]
    
    def get_active_constraints(self) -> List[Constraint]:
        """Get all active constraints"""
        return [c for c in self.constraints.values() if c.is_active]
    
    def deactivate_constraint(self, constraint_name: str):
        """Temporarily deactivate a constraint"""
        if constraint_name in self.constraints:
            self.constraints[constraint_name].is_active = False
    
    def activate_constraint(self, constraint_name: str):
        """Reactivate a constraint"""
        if constraint_name in self.constraints:
            self.constraints[constraint_name].is_active = True
    
    def copy(self) -> 'Model':
        """Create a deep copy of the model"""
        import copy
        return copy.deepcopy(self)


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


class SolverInterface(ABC):
    """Abstract interface for optimization solvers"""
    
    @abstractmethod
    def load_model(self, model: Model):
        """Load a model into the solver"""
        pass
    
    @abstractmethod
    def solve(self) -> SolutionStatus:
        """Solve the current model"""
        pass
    
    @abstractmethod
    def get_solution(self) -> Optional[Dict[str, float]]:
        """Get the solution values for variables"""
        pass
    
    @abstractmethod
    def get_objective_value(self) -> Optional[float]:
        """Get the objective function value"""
        pass
    
    @abstractmethod
    def get_dual_values(self) -> Optional[Dict[str, float]]:
        """Get dual values (shadow prices) for constraints"""
        pass
    
    @abstractmethod
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the current model"""
        pass
    
    @abstractmethod
    def remove_constraint(self, constraint_name: str):
        """Remove a constraint from the current model"""
        pass
    
    @abstractmethod
    def set_constraint_active(self, constraint_name: str, active: bool):
        """Activate or deactivate a constraint"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the solver state"""
        pass
    
    @abstractmethod
    def set_time_limit(self, seconds: float):
        """Set solver time limit"""
        pass
    
    @abstractmethod
    def set_verbose(self, verbose: bool):
        """Set solver verbosity"""
        pass


class IISAlgorithm(ABC):
    """Abstract base class for IIS finding algorithms"""
    
    def __init__(self, solver: SolverInterface):
        self.solver = solver
        self.model: Optional[Model] = None
        
    @abstractmethod
    def find_iis(self, model: Model) -> IISResult:
        """Find an IIS in the given model"""
        pass
    
    def verify_iis(self, model: Model, iis_constraints: Set[str]) -> bool:
        """
        Verify that a set of constraints forms an IIS
        
        Returns:
            True if the constraints form a valid IIS
        """
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