"""
HiGHS Solver Interface Implementation
"""

import highspy
import numpy as np
from typing import Dict, Optional, List
from collections import defaultdict
from core.solver_interface import SolverInterface
from core.base import (
	Model, Variable, Constraint, 
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

	def reset(self):
		"""Reset the solver state"""
		self.highs.clear()
		self.highs = highspy.Highs()
		self.model = None
		self.var_indices = {}
		self.constraint_indices = {}
		self.solution = None
		self.dual_values = None

	def load_model(self, model: Model):
		"""Load a model into HiGHS solver"""
		self.reset()
		self.model = model

		# Configure solver options
		self.highs.setOptionValue("output_flag", str(self.verbose).lower())
		self.highs.setOptionValue("presolve", "off")

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
		
		if active_constraints:
			# Build constraint in CSC format
			row_indices: Dict[int, List[int]] = defaultdict(list)		# list of row indices for each column for nz CSC format
			col_indices = []
			values: Dict[int, List[float]] = defaultdict(list)
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
						row_indices[var_idx].append(idx)
						values[var_idx].append(coeff)
						col_indices.append(self.var_indices[var_name])
						values.append(coeff)

		# Add variables
		# self.highs.addVars(
		# 	num_vars=num_vars,
		# 	lower_bounds=lower_bounds,
		# 	upper_bounds=upper_bounds,
		# )

		self.highs.addCols(
			num_cols=num_vars,
			costs=obj_coeffs,
			lower_bounds=lower_bounds,
			upper_bounds=upper_bounds,
			num_elements=len(values),
			starts=0,
			indices=None,
			values=None,
		)


		



