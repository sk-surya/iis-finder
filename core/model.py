


from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.base import Constraint, Objective, Variable



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