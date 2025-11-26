	

from abc import ABC, abstractmethod
from typing import Dict, Optional

from core.base import Constraint, Model, SolutionStatus


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

    @abstractmethod
    def sync_model(self, model: Model):
        """
        Sync model changes incrementally if possible.
        Falls back to full load_model() if incremental not possible.

        This is the preferred method for updating an already-loaded model.
        """
        pass

    @abstractmethod
    def supports_incremental_updates(self) -> bool:
        """
        Check whether this solver supports incremental constraint updates.

        Returns:
            True if solver can apply incremental updates, False otherwise
        """
        pass