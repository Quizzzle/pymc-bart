"""Decision Table (symmetric tree) implementation for CatBoost-style trees."""

import numpy as np
import numpy.typing as npt
from pytensor import config

from .split_rules import SplitRule


class DecisionTableNode:
    """Node in a Decision Table (symmetric tree).
    
    Attributes
    ----------
    value : npt.NDArray
        Leaf node value or split value
    idx_split_variable : int
        Index of the split variable (-1 for leaf nodes)
    children : dict[int, DecisionTableNode]
        Child nodes indexed by split outcome
    nvalue : int
        Number of data points in this node
    linear_params : Optional[list[npt.NDArray]]
        Linear regression parameters for leaf nodes
    """

    __slots__ = ("value", "idx_split_variable", "children", "nvalue", "linear_params")

    def __init__(
        self,
        value: npt.NDArray = np.array([-1.0]),
        idx_split_variable: int = -1,
        children: dict | None = None,
        nvalue: int = 0,
        linear_params: list[npt.NDArray] | None = None,
    ) -> None:
        self.value = value
        self.idx_split_variable = idx_split_variable
        self.children = children or {}
        self.nvalue = nvalue
        self.linear_params = linear_params

    def is_leaf_node(self) -> bool:
        return self.idx_split_variable < 0

    def is_split_node(self) -> bool:
        return not self.is_leaf_node()


class DecisionTable:
    """Decision Table (symmetric tree) - CatBoost style symmetric tree.
    
    A Decision Table is a tree where all nodes at the same depth use the same split variable.
    This creates a more interpretable structure compared to standard binary trees.
    
    Attributes
    ----------
    root : DecisionTableNode
        Root node of the decision table
    split_rules : list[SplitRule]
        Split rules for each variable
    depth : int
        Maximum depth of the table
    output : npt.NDArray
        Predictions array
    """

    __slots__ = (
        "root",
        "split_rules",
        "depth",
        "output",
        "level_variables",
        "level_split_values",
    )

    def __init__(
        self,
        root: DecisionTableNode,
        split_rules: list[SplitRule],
        output: npt.NDArray,
        level_variables: list[int] | None = None,
        level_split_values: list[npt.NDArray] | None = None,
    ) -> None:
        self.root = root
        self.split_rules = split_rules
        self.output = output
        self.level_variables = level_variables or []
        self.level_split_values = level_split_values or []
        self.depth = self._compute_depth()
        self._refresh_metadata()

    @classmethod
    def new_decision_table(
        cls,
        leaf_node_value: npt.NDArray,
        num_observations: int,
        shape: int,
        split_rules: list[SplitRule],
    ) -> "DecisionTable":
        """Create a new Decision Table with a single leaf node."""
        return cls(
            root=DecisionTableNode(
                value=leaf_node_value,
                idx_split_variable=-1,
                nvalue=num_observations,
            ),
            split_rules=split_rules,
            output=np.zeros((num_observations, shape)).astype(config.floatX),
            level_variables=[],
            level_split_values=[],
        )

    def _compute_depth(self) -> int:
        """Compute the depth of the decision table."""
        def _depth_recursive(node: DecisionTableNode) -> int:
            if node.is_leaf_node():
                return 0
            return 1 + max(_depth_recursive(child) for child in node.children.values())

        return _depth_recursive(self.root)

    def is_symmetric(self) -> bool:
        """Check if the tree is symmetric (all nodes at same depth use same split variable)."""
        def _check_symmetry_recursive(node: DecisionTableNode, expected_var: int | None = None) -> tuple[bool, int | None]:
            if node.is_leaf_node():
                return True, expected_var

            split_var = node.idx_split_variable

            if expected_var is not None and split_var != expected_var:
                return False, expected_var

            for child in node.children.values():
                is_sym, next_var = _check_symmetry_recursive(child, split_var)
                if not is_sym:
                    return False, next_var

            return True, split_var

        is_sym, _ = _check_symmetry_recursive(self.root)
        return is_sym

    def grow_leaf_node(
        self,
        leaf_node: DecisionTableNode,
        selected_predictor: int,
        split_value: npt.NDArray,
        left_value: npt.NDArray,
        right_value: npt.NDArray,
        left_nvalue: int,
        right_nvalue: int,
        depth: int,
    ) -> None:
        """Grow a leaf node by creating two child nodes."""
        split_var, split_threshold = self._resolve_level_predicate(
            depth, selected_predictor, split_value
        )

        leaf_node.idx_split_variable = split_var
        leaf_node.value = split_threshold.copy()

        leaf_node.children[0] = DecisionTableNode(
            value=left_value,
            idx_split_variable=-1,
            nvalue=left_nvalue,
        )
        leaf_node.children[1] = DecisionTableNode(
            value=right_value,
            idx_split_variable=-1,
            nvalue=right_nvalue,
        )
        leaf_node.nvalue = left_nvalue + right_nvalue
        self._refresh_metadata()

    def predict(
        self,
        X: npt.NDArray,
        excluded: list[int] | None = None,
        shape: int = 1,
    ) -> npt.NDArray:
        """
        Predict output for given input data.
        
        Parameters
        ----------
        X : npt.NDArray
            Input data
        excluded : Optional[list[int]]
            Variables to exclude from prediction
        shape : int
            Output shape
            
        Returns
        -------
        npt.NDArray
            Predictions
        """
        if excluded is None:
            excluded = []

        x_shape = (1,) if len(X.shape) == 1 else X.shape[:-1]
        nd_dims = (...,) + (None,) * len(x_shape)

        p_d = (
            np.zeros(shape + x_shape) if isinstance(shape, tuple) else np.zeros((shape,) + x_shape)
        )

        # Traverse the decision table
        def _traverse(node: DecisionTableNode, weights: npt.NDArray, split_var_parent: int = -1):
            if node.is_leaf_node():
                params = node.linear_params
                if params is None:
                    p_d_leaf = weights * node.value[nd_dims]
                else:
                    p_d_leaf = weights * (
                        params[0][nd_dims] + params[1][nd_dims] * X[..., split_var_parent]
                    )
                return p_d_leaf
            else:
                split_var = node.idx_split_variable
                result = np.zeros_like(p_d)

                if excluded is not None and split_var in excluded:
                    # Average over both branches
                    for child_idx, child in node.children.items():
                        prop = child.nvalue / node.nvalue
                        result += _traverse(child, weights * prop, split_var)
                else:
                    # Split based on split rule
                    to_left = (
                        self.split_rules[split_var]
                        .divide(X[..., split_var], node.value)
                        .astype("float")
                    )
                    
                    if 0 in node.children:
                        result += _traverse(node.children[0], weights * to_left, split_var)
                    if 1 in node.children:
                        result += _traverse(node.children[1], weights * (1 - to_left), split_var)

                return result

        p_d = _traverse(self.root, np.ones(x_shape), -1)

        if len(X.shape) == 1:
            p_d = p_d[..., 0]

        return p_d

    def get_leaf_nodes(self, with_depth: bool = False) -> list:
        """Get all leaf nodes in the decision table."""
        leaf_nodes = []

        for node, depth in self._traverse_with_depth():
            if node.is_leaf_node():
                if with_depth:
                    leaf_nodes.append((node, depth))
                else:
                    leaf_nodes.append(node)

        return leaf_nodes

    def get_split_nodes(self, with_depth: bool = False) -> list:
        """Get all split nodes in the decision table."""
        split_nodes = []

        for node, depth in self._traverse_with_depth():
            if node.is_split_node():
                if with_depth:
                    split_nodes.append((node, depth))
                else:
                    split_nodes.append(node)

        return split_nodes

    def get_level_predicate(
        self, depth: int
    ) -> tuple[int | None, npt.NDArray | None]:
        """Return (variable, split_value) for the provided depth, if defined."""
        if depth < 0:
            return None, None
        if depth >= len(self.level_variables):
            return None, None

        split_var = self.level_variables[depth]
        if split_var < 0:
            return None, None

        split_value = (
            self.level_split_values[depth].copy()
            if depth < len(self.level_split_values)
            else None
        )
        return split_var, split_value

    def copy(self) -> "DecisionTable":
        """Create a deep copy of the decision table."""
        def _copy_node(node: DecisionTableNode) -> DecisionTableNode:
            return DecisionTableNode(
                value=node.value.copy(),
                idx_split_variable=node.idx_split_variable,
                children={k: _copy_node(v) for k, v in node.children.items()},
                nvalue=node.nvalue,
                linear_params=[p.copy() for p in node.linear_params] if node.linear_params else None,
            )

        return DecisionTable(
            root=_copy_node(self.root),
            split_rules=self.split_rules,
            output=self.output.copy(),
            level_variables=self.level_variables.copy(),
            level_split_values=[val.copy() for val in self.level_split_values],
        )

    def trim(self) -> "DecisionTable":
        """Create a trimmed copy without data point indices."""
        def _trim_node(node: DecisionTableNode) -> DecisionTableNode:
            return DecisionTableNode(
                value=node.value.copy(),
                idx_split_variable=node.idx_split_variable,
                children={k: _trim_node(v) for k, v in node.children.items()},
                nvalue=node.nvalue,
                linear_params=[p.copy() for p in node.linear_params] if node.linear_params else None,
            )

        return DecisionTable(
            root=_trim_node(self.root),
            split_rules=self.split_rules,
            output=np.array([-1]),
            level_variables=self.level_variables.copy(),
            level_split_values=[val.copy() for val in self.level_split_values],
        )

    def prune_node(self, node: DecisionTableNode, new_value: npt.NDArray, nvalue: int) -> None:
        """Convert a split node into a leaf node and refresh metadata."""
        node.idx_split_variable = -1
        node.value = new_value
        node.children = {}
        node.nvalue = nvalue
        self._refresh_metadata()

    def update_level_predicate(
        self, depth: int, split_variable: int, split_value: npt.NDArray
    ) -> None:
        """Update predicate (variable + threshold) for a specific depth."""
        for node, node_depth in self._traverse_with_depth():
            if node_depth == depth and node.is_split_node():
                node.idx_split_variable = split_variable
                node.value = split_value.copy()

        self._ensure_level_capacity(depth)
        self.level_variables[depth] = split_variable
        self.level_split_values[depth] = split_value.copy()
        self._refresh_metadata()

    def count_split_nodes(self) -> int:
        """Return number of split nodes."""
        return sum(1 for node, _ in self._traverse_with_depth() if node.is_split_node())

    def count_leaf_nodes(self) -> int:
        """Return number of leaf nodes."""
        return sum(1 for node, _ in self._traverse_with_depth() if node.is_leaf_node())

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _refresh_metadata(self) -> None:
        """Update cached depth and level predicates."""
        self.depth = self._compute_depth()
        inferred_vars, inferred_values = self._infer_level_metadata()
        self.level_variables = inferred_vars
        self.level_split_values = inferred_values

    def _infer_level_metadata(self) -> tuple[list[int], list[npt.NDArray]]:
        """Infer level-wise predicates from the current tree structure."""
        level_vars: dict[int, int] = {}
        level_values: dict[int, npt.NDArray] = {}

        for node, depth in self._traverse_with_depth():
            if node.is_split_node():
                if depth in level_vars:
                    if level_vars[depth] != node.idx_split_variable:
                        raise ValueError(
                            f"Inconsistent split variable at depth {depth}. "
                            "Decision table must remain symmetric."
                        )
                    if not np.array_equal(level_values[depth], node.value):
                        raise ValueError(
                            f"Inconsistent split value at depth {depth}. "
                            "Decision table must reuse identical predicates per level."
                        )
                else:
                    level_vars[depth] = node.idx_split_variable
                    level_values[depth] = node.value.copy()

        if not level_vars:
            return [], []

        depths = sorted(level_vars.keys())
        return (
            [level_vars[d] for d in depths],
            [level_values[d] for d in depths],
        )

    def _traverse_with_depth(self):
        """Yield nodes with their respective depth."""
        stack = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            yield node, depth
            for child in node.children.values():
                stack.append((child, depth + 1))

    def _resolve_level_predicate(
        self, depth: int, split_variable: int, split_value: npt.NDArray
    ) -> tuple[int, npt.NDArray]:
        """Ensure predicate consistency at a given depth."""
        if depth < len(self.level_variables):
            return self.level_variables[depth], self.level_split_values[depth]

        if split_value is None:
            raise ValueError("Split value must be provided for new depth levels.")

        self._ensure_level_capacity(depth)
        self.level_variables[depth] = split_variable
        self.level_split_values[depth] = split_value.copy()
        return split_variable, split_value

    def _ensure_level_capacity(self, depth: int) -> None:
        """Ensure the metadata arrays can store information for the given depth."""
        while len(self.level_variables) <= depth:
            self.level_variables.append(-1)
        while len(self.level_split_values) <= depth:
            self.level_split_values.append(np.array([-1.0]))
