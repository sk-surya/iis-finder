"""
HiGHS Solver Interface Implementation
"""

import highspy
import numpy as np
from typing import Dict, Optional, List, override
from collections import defaultdict
from core.solver_interface import SolverInterface
from core.model import Model
from core.base import (
	ModelChange, Variable, Constraint,
	SolutionStatus, ConstraintType, VariableType
)


class HighsSolver(SolverInterface):
	""""HiGHS solver implementation of the SolverInterface"""

	def __init__(self):
		self.highs = highspy.Highs()
		self.model: Optional[Model] = None
		self.var_indices: Dict[str, int] = {}
		self.constraint_indices: Dict[str, int] = {}
		self.solution: Optional[Dict[str, float]] = None
		self.dual_values: Optional[Dict[str, float]] = None
		self.verbose = False
		self.synced_model_version: int = -1

	@override
	def reset(self):
		"""Reset the solver state"""
		self.highs.clear()
		self.highs = highspy.Highs()
		self.model = None
		self.var_indices = {}
		self.constraint_indices = {}
		self.solution = None
		self.dual_values = None
		self.synced_model_version = -1

	@override
	def load_model(self, model: Model):
		"""Load a model into HiGHS solver"""
		self.reset()
		self.model = model

		# Configure solver options
		# self.highs.setOptionValue("output_flag", str(self.verbose).lower())
		# self.highs.setOptionValue("presolve", "off")

		# Create variable index mapping
		var_list = sorted(model.variables.keys())
		self.var_indices = {var_name: idx for idx, var_name in enumerate(var_list)}
		num_vars = len(var_list)

		# Set up variables
		lower_bounds = []
		upper_bounds = []
		integrality = []

		for var_name in var_list:
			var = model.variables[var_name]
			lower_bounds.append(var.lower_bound)
			upper_bounds.append(var.upper_bound)
			if var.var_type == VariableType.BINARY:
				integrality.append(highspy.HighsVarType.kInteger)
				lower_bounds[-1] = 0.0
				upper_bounds[-1] = 1.0
			elif var.var_type == VariableType.INTEGER:
				integrality.append(highspy.HighsVarType.kInteger)
			else:
				integrality.append(highspy.HighsVarType.kContinuous)

		# Set up objective
		obj_coeffs = np.zeros(num_vars)
		if model.objective:
			for var_name, coef in model.objective.coefficients.items():
				if var_name in self.var_indices:
					obj_coeffs[self.var_indices[var_name]] = coef

		# Set up constraints
		active_constraints = model.get_active_constraints()
		self.constraint_indices = {constraint.name: idx for idx, constraint in enumerate(active_constraints)}
		
		counts = []
		if active_constraints:
			# Build constraint in CSC format
			# row_indices: Dict[int, List[int]] = defaultdict(list)		# list of row indices for each column for nz CSC format
			col_indices = []
			# values: Dict[int, List[float]] = defaultdict(list)
			values: List[float] = []
			lhs = []
			rhs = []

			for idx, constraint in enumerate(active_constraints):
				# Set bounds based on constraint type
				if constraint.constraint_type == ConstraintType.LEQ:
					lhs.append(-highspy.kHighsInf)
					rhs.append(constraint.rhs)
				elif constraint.constraint_type == ConstraintType.GEQ:
					lhs.append(constraint.rhs)
					rhs.append(highspy.kHighsInf)
				else:  # EQ
					lhs.append(constraint.rhs)
					rhs.append(constraint.rhs)

				# Add non-zero coefficients
				for var_name, coeff in constraint.coefficients.items():
					if var_name in self.var_indices and coeff != 0:
						var_idx = self.var_indices[var_name]
						# row_indices[var_idx].append(idx)
						# values[var_idx].append(coeff)
						values.append(coeff)
						col_indices.append(var_idx)
				counts.append(len(constraint.coefficients))

		# Add variables
		# self.highs.addVars(
		# 	num_vars=num_vars,
		# 	lower_bounds=lower_bounds,
		# 	upper_bounds=upper_bounds,
		# )

		# values_list_ordered_by_var = []
		# row_indices_list_ordered_by_var = []
		# counts = []
		# for var_idx in self.var_indices.values():
		# 	values_list_ordered_by_var.extend(values[var_idx])
		# 	row_indices_list_ordered_by_var.extend(row_indices[var_idx])
		# 	counts.append(len(values[var_idx]))
		
		# Build starts array for CSC format (length = num_vars, not num_vars + 1)
		if counts:
			starts = [0] + np.cumsum(counts[:-1]).tolist()
		else:
			starts = [0] * num_vars

		# addCols expects positional arguments, not keyword arguments
		# Args: num_cols, costs, lower, upper, num_nz, starts (int32), indices (int32), values (float64)
		# status = self.highs.addCols(
		# 	num_vars,
		# 	obj_coeffs,
		# 	np.array(lower_bounds),
		# 	np.array(upper_bounds),
		# 	len(values_list_ordered_by_var),
		# 	np.array(starts, dtype=np.int32),
		# 	np.array(row_indices_list_ordered_by_var, dtype=np.int32),
		# 	np.array(values_list_ordered_by_var, dtype=np.float64),
		# )
		
		# inf = highspy.kHighsInf
		# The constraint matrix is defined with the rows below, but parameters
		# for an empty (column-wise) matrix must be passed
		status = self.highs.addCols(
			num_vars,
			obj_coeffs,
			np.array(lower_bounds),
			np.array(upper_bounds),
			0,
			0,
			0,
			0
		)

		assert status == highspy.HighsStatus.kOk, "Error adding variables to HiGHS model"
		# Add the rows, with the constraint matrix row-wise
		if active_constraints:
			status = self.highs.addRows(
				len(active_constraints),
				np.array(lhs),
				np.array(rhs),
				len(values),
				np.array(starts, dtype=np.int32),
				np.array(col_indices, dtype=np.int32),
				np.array(values, dtype=np.float64)
			)

		# Add integrality of vars
		self.highs.changeColsIntegrality(
			num_vars,
			np.arange(num_vars, dtype=np.int32),
			np.array(integrality, dtype=np.uint8)
		)

		# Set objective sense
		if model.objective and not model.objective.is_minimize:
			self.highs.changeObjectiveSense(highspy.ObjSense.kMaximize)

		# Mark model as synced
		self.synced_model_version = model.get_version()

	@override
	def solve(self) -> SolutionStatus:
		"""Solve the loaded model"""
		if self.synced_model_version != self.model.get_version():
			print(f"Warning - Model (version={self.model.get_version()}) is not synced with solver's model (version={self.synced_model_version}). Auto-syncing now.")
			self.sync_model(self.model)
		self.highs.run()
		status = self.highs.getModelStatus()
		
		# Map HiGHS status to SolutionStatus
		if status == highspy.HighsModelStatus.kOptimal:
			self._extract_solution()
			return SolutionStatus.OPTIMAL
		elif status == highspy.HighsModelStatus.kInfeasible:
			return SolutionStatus.INFEASIBLE
		elif status == highspy.HighsModelStatus.kUnbounded:
			return SolutionStatus.UNBOUNDED
		elif status == highspy.HighsModelStatus.kTimeLimit:
			return SolutionStatus.TIME_LIMIT
		else:
			return SolutionStatus.UNKNOWN
	
	def _extract_solution(self):
		"""Extract solution and dual values from HiGHS"""
		solution = self.highs.getSolution()
		col_values = list(solution.col_value)
		dual_values = list(solution.row_dual)

		# Extract variable values
		self.solution = {}
		for var_name, idx in self.var_indices.items():
			self.solution[var_name] = col_values[idx]
		
		# Extract dual values for constraints
		self.dual_values = {}
		assert self.model is not None, "Model must be loaded before extracting solution"
		active_constraints = self.model.get_active_constraints()
		for constraint in active_constraints:
			idx = self.constraint_indices[constraint.name]
			self.dual_values[constraint.name] = dual_values[idx]

	@override
	def get_solution(self) -> Optional[Dict[str, float]]:
		"""Get the solution values for variables"""
		return self.solution

	@override
	def get_objective_value(self) -> Optional[float]:
		"""Get the objective function value"""
		info = self.highs.getInfo()
		return info.objective_function_value if info else None
	
	@override
	def get_dual_values(self) -> Optional[Dict[str, float]]:
		"""Get dual values (shadow prices) for constraints"""
		return self.dual_values
	
	@override
	def add_constraint(self, constraint: Constraint):
		"""Add a constraint to the current model"""
		if self.model is None:
			raise ValueError("Model must be loaded before adding constraints")
		self.model.add_constraint(constraint)
		self.sync_model(self.model)

	@override
	def remove_constraint(self, constraint_name: str):
		"""Remove a constraint from the current model"""
		if self.model is None:
			raise ValueError("Model must be loaded before removing constraints")
		self.model.remove_constraint(constraint_name)
		self.sync_model(self.model)

	@override
	def set_constraint_active(self, constraint_name: str, active: bool):
		"""Activate or deactivate a constraint"""
		if self.model is None:
			raise ValueError("Model must be loaded before modifying constraints")
		if active:
			self.model.activate_constraint(constraint_name)
		else:
			self.model.deactivate_constraint(constraint_name)
		self.sync_model(self.model)

	@override
	def set_time_limit(self, seconds: float):
		"""Set solver time limit"""
		self.highs.setOptionValue("time_limit", seconds)

	@override
	def set_verbose(self, verbose: bool):
		self.verbose = verbose
		self.highs.setOptionValue("output_flag", str(self.verbose).lower())

	@override
	def supports_incremental_updates(self) -> bool:
		"""HiGHS supports incremental constraint updates"""
		return True

	@override
	def sync_model(self, model: Model):
		"""Sync model changes incrementally if possible"""
		from core.base import ChangeType

		if self.model is None or self.synced_model_version == -1:
			# First load or no previous model
			self.load_model(model)
			return

		# Get changes since last sync
		changes = model.get_changes_since(self.synced_model_version)

		if not changes:
			# No changes, nothing to do
			return

		# Check if incremental update is feasible
		if self._should_full_reload(changes):
			self.load_model(model)
			return

		# Apply incremental changes
		self.model = model
		for change in changes:
			self._apply_change(change)

		self.synced_model_version = model.get_version()

	def _should_full_reload(self, changes: List['ModelChange']) -> bool:
		"""Determine if changes require full reload"""
		from core.base import ChangeType
		# Variable additions require full reload (affects index mapping)
		# Objective changes might require full reload
		for change in changes:
			if change.change_type in [
				ChangeType.VARIABLE_ADDED,
				ChangeType.OBJECTIVE_CHANGED
			]:
				return True

		# Heuristic: if many changes, might be faster to full reload
		# For now, always try incremental for constraint-only changes
		return False

	def _apply_change(self, change: 'ModelChange'):
		"""Apply a single change incrementally"""
		from core.base import ChangeType
		if change.change_type == ChangeType.CONSTRAINT_ADDED:
			self._add_constraint_incremental(change.entity)
		elif change.change_type == ChangeType.CONSTRAINT_REMOVED:
			self._remove_constraint_incremental(change.entity_name)
		elif change.change_type == ChangeType.CONSTRAINT_ACTIVATED:
			self._activate_constraint_incremental(change.entity_name)
		elif change.change_type == ChangeType.CONSTRAINT_DEACTIVATED:
			self._deactivate_constraint_incremental(change.entity_name)

	def _deactivate_constraint_incremental(self, constraint_name: str):
		"""Deactivate a constraint by relaxing bounds to (-inf, inf)"""
		if constraint_name not in self.constraint_indices:
			return

		row_idx = self.constraint_indices[constraint_name]
		# Set bounds to (-inf, inf) - always satisfied
		self.highs.changeRowBounds(row_idx, -highspy.kHighsInf, highspy.kHighsInf)

		# Clear cached solution/duals as they may be invalid
		self.solution = None
		self.dual_values = None

	def _activate_constraint_incremental(self, constraint_name: str):
		"""Reactivate a deactivated constraint by restoring original bounds"""
		if self.model is None:
			raise ValueError("Model must be loaded before activating constraints")

		constraint = self.model.constraints[constraint_name]

		if constraint_name not in self.constraint_indices:
			# Constraint was fully removed, need to add it back
			self._add_constraint_incremental(constraint)
		else:
			# Constraint exists, just restore bounds
			row_idx = self.constraint_indices[constraint_name]
			if constraint.constraint_type == ConstraintType.LEQ:
				lhs = -highspy.kHighsInf
				rhs = constraint.rhs
			elif constraint.constraint_type == ConstraintType.GEQ:
				lhs = constraint.rhs
				rhs = highspy.kHighsInf
			else:  # EQ
				lhs = constraint.rhs
				rhs = constraint.rhs

			self.highs.changeRowBounds(row_idx, lhs, rhs)

			# Clear cached solution/duals as they may be invalid
			self.solution = None
			self.dual_values = None

	def _add_constraint_incremental(self, constraint: Constraint):
		"""Add a single constraint using HiGHS addRows API"""
		# Build constraint bounds
		if constraint.constraint_type == ConstraintType.LEQ:
			lhs = -highspy.kHighsInf
			rhs = constraint.rhs
		elif constraint.constraint_type == ConstraintType.GEQ:
			lhs = constraint.rhs
			rhs = highspy.kHighsInf
		else:  # EQ
			lhs = constraint.rhs
			rhs = constraint.rhs

		# Build sparse row representation
		indices = []
		values = []
		for var_name, coeff in constraint.coefficients.items():
			if var_name in self.var_indices and coeff != 0:
				indices.append(self.var_indices[var_name])
				values.append(coeff)

		# Add row to HiGHS
		# addRows args: num_rows, lower, upper, num_nz, starts (int32), indices (int32), values (float64)
		self.highs.addRows(
			1,
			np.array([lhs]),
			np.array([rhs]),
			len(values),
			np.array([0], dtype=np.int32),
			np.array(indices, dtype=np.int32),
			np.array(values, dtype=np.float64)
		)

		# Update constraint index mapping
		# New row is added at the end
		new_row_idx = len(self.constraint_indices)
		self.constraint_indices[constraint.name] = new_row_idx

		# Clear cached solution/duals as they may be invalid
		self.solution = None
		self.dual_values = None

	def _remove_constraint_incremental(self, constraint_name: str):
		"""Remove a constraint using HiGHS deleteRows API"""
		if constraint_name not in self.constraint_indices:
			return

		row_idx = self.constraint_indices[constraint_name]

		# Delete row from HiGHS
		# deleteRows args: num_rows, rows (int32 array)
		self.highs.deleteRows(
			1,
			np.array([row_idx], dtype=np.int32)
		)

		# Update index mapping - all constraints after this shift down
		del self.constraint_indices[constraint_name]
		for name, idx in list(self.constraint_indices.items()):
			if idx > row_idx:
				self.constraint_indices[name] = idx - 1

		# Clear cached solution/duals as they may be invalid
		self.solution = None
		self.dual_values = None

	def load_from_file(self, filename: str):
		"""Load a model from a file (MPS or LP format)"""
		self.reset()
		self.highs.readModel(filename)

		# Extract model structure and create Model object
		lp = self.highs.getLp()
		model = Model(name=filename.split('/')[-1].split('.')[0])

		# Create variables
		for i in range(lp.num_col_):
			var_type = VariableType.CONTINUOUS
			if hasattr(lp, 'integrality_') and lp.integrality_:
				if lp.integrality_[i] == highspy.HighsVarType.kInteger:
					var_type = VariableType.INTEGER
					if lp.col_lower_[i] == 0.0 and lp.col_upper_[i] == 1.0:
						var_type = VariableType.BINARY
			var = Variable(
				name=f"x{i}",
				var_type=var_type,
				lower_bound=lp.col_lower_[i],
				upper_bound=lp.col_upper_[i],
				index=i
			)
			model.add_variable(var)

		# Create constraints
	
		if lp.a_matrix_.format_ == highspy.MatrixFormat.kColwise:
			starts = list(lp.a_matrix_.start_)
			coeffs = list(lp.a_matrix_.value_)
			indices = list(lp.a_matrix_.index_)

			for col_idx in range(lp.num_col_):
				col_slice = slice(starts[col_idx], starts[col_idx + 1])
				col_coeffs = coeffs[col_slice]
				col_coeff_row_indices = indices[col_slice]
				for i, row_idx in enumerate(col_coeff_row_indices):
					constraint_name = f"c{row_idx}"
					if constraint_name not in model.constraints:
						# Determine constraint type and rhs
						if lp.row_lower_[row_idx] == lp.row_upper_[row_idx]:
							constraint_type = ConstraintType.EQ
							rhs = lp.row_upper_[row_idx]
						elif lp.row_lower_[row_idx] > -highspy.kHighsInf:
							constraint_type = ConstraintType.GEQ
							rhs = lp.row_lower_[row_idx]
						else:
							constraint_type = ConstraintType.LEQ
							rhs = lp.row_upper_[row_idx]
						
						constraint = Constraint(
							name=constraint_name,
							coefficients={},
							constraint_type=constraint_type,
							rhs=rhs,
							index=row_idx
						)
						model.add_constraint(constraint)
					
					var_name = f"x{col_idx}"
					model.constraints[constraint_name].coefficients[var_name] = col_coeffs[i]
		
		# TODO as test: check if highs model loaded from this model is same as self.highs model
		return model



		



