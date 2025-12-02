


from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.base import Constraint, Objective, Variable, ModelChange, ChangeType
else:
    # Import at runtime to avoid circular import
    from core.base import Constraint, Objective, Variable



@dataclass
class Model:
    """Represents an optimization model with change tracking"""
    name: str = "model"
    variables: Dict[str, Variable] = field(default_factory=dict)
    constraints: Dict[str, Constraint] = field(default_factory=dict)
    objective: Optional[Objective] = None
    _changes: List['ModelChange'] = field(default_factory=list)
    _version: int = 0
    
    def add_variable(self, variable: Variable):
        """Add a variable to the model and track the change"""
        from core.base import ModelChange, ChangeType
        self.variables[variable.name] = variable
        self._changes.append(ModelChange(
            change_type=ChangeType.VARIABLE_ADDED,
            entity_name=variable.name,
            entity=variable
        ))
        self._version += 1

    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the model and track the change"""
        from core.base import ModelChange, ChangeType
        self.constraints[constraint.name] = constraint
        self._changes.append(ModelChange(
            change_type=ChangeType.CONSTRAINT_ADDED,
            entity_name=constraint.name,
            entity=constraint
        ))
        self._version += 1

    def remove_constraint(self, constraint_name: str):
        """Remove a constraint from the model and track the change"""
        from core.base import ModelChange, ChangeType
        if constraint_name in self.constraints:
            del self.constraints[constraint_name]
            self._changes.append(ModelChange(
                change_type=ChangeType.CONSTRAINT_REMOVED,
                entity_name=constraint_name
            ))
            self._version += 1

    def get_active_constraints(self) -> List[Constraint]:
        """Get all active constraints"""
        return [c for c in self.constraints.values() if c.is_active]

    def deactivate_constraint(self, constraint_name: str):
        """Temporarily deactivate a constraint and track the change"""
        from core.base import ModelChange, ChangeType
        if constraint_name in self.constraints:
            self.constraints[constraint_name].is_active = False
            self._changes.append(ModelChange(
                change_type=ChangeType.CONSTRAINT_DEACTIVATED,
                entity_name=constraint_name
            ))
            self._version += 1

    def activate_constraint(self, constraint_name: str):
        """Reactivate a constraint and track the change"""
        from core.base import ModelChange, ChangeType
        if constraint_name not in self.constraints:
             raise ValueError(f"Constraint '{constraint_name}' does not exist in the model.")
        self.constraints[constraint_name].is_active = True
        self._changes.append(ModelChange(
            change_type=ChangeType.CONSTRAINT_ACTIVATED,
            entity_name=constraint_name
        ))
        self._version += 1

    def get_changes_since(self, version: int) -> List['ModelChange']:
        """Get all changes since a specific version"""
        if version < 0:
            return self._changes.copy()
        # Return changes that were added after the specified version
        # Since version increments with each change, we need changes from version+1 onwards
        num_changes_at_version = version
        return self._changes[num_changes_at_version:]

    def clear_changes(self):
        """Clear the change log (called by solver after sync)"""
        self._changes.clear()

    def get_version(self) -> int:
        """Get current model version"""
        return self._version

    def copy(self) -> 'Model':
        """Create a deep copy of the model with cleared change log"""
        import copy
        model_copy = copy.deepcopy(self)
        # Clear change log on copy - treat as fresh model
        model_copy._changes.clear()
        return model_copy