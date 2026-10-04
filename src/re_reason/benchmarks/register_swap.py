"""Procedural synthetic benchmark for Register Tracking (Phase 1).

Tracks the integer values of 4 registers across 10 sequential assignments,
swaps, and modular additions.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from re_reason.benchmarks.base import BaseTask, TaskBatch


class RegisterTrackingTask(BaseTask):
    """Procedurally generated Register Tracking task.

    State: 4 integer registers (R0, R1, R2, R3) taking values in {0, ..., num_values - 1}.
    Horizon: sequence_length (default 10) sequential operations.

    Operations:
        0: SET reg_a = val
        1: SWAP reg_a, reg_b
        2: ADD reg_a = (reg_a + reg_b) % num_values

    Input representation at step t:
        One-hot vector of length:
            3 (op_type) + num_registers (reg_a) + num_registers (reg_b) + num_values (immediate)
        Default: 3 + 4 + 4 + 10 = 21 dimensions.
    """

    OP_SET = 0
    OP_SWAP = 1
    OP_ADD = 2
    NUM_OPS = 3

    def __init__(
        self,
        num_registers: int = 4,
        sequence_length: int = 10,
        num_values: int = 10,
        seed: Optional[int] = None,
        op_weights: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.num_registers = num_registers
        self.sequence_length = sequence_length
        self.num_values = num_values
        self.input_dim = self.NUM_OPS + self.num_registers + self.num_registers + self.num_values
        self.op_weights = None
        if op_weights is not None:
            weights = np.array(op_weights, dtype=np.float64)
            self.op_weights = (weights / weights.sum()).tolist()
        self._rng = np.random.default_rng(seed if seed is not None else 42)

    def _encode_operation(self, op_type: int, reg_a: int, reg_b: int, val: int) -> np.ndarray:
        """Encodes an operation into a fixed-length 1D float vector."""
        vec = np.zeros(self.input_dim, dtype=np.float32)
        # Op type (0..2)
        vec[op_type] = 1.0

        offset = self.NUM_OPS
        # reg_a (0..num_registers-1)
        if 0 <= reg_a < self.num_registers:
            vec[offset + reg_a] = 1.0

        offset += self.num_registers
        # reg_b (0..num_registers-1)
        if 0 <= reg_b < self.num_registers:
            vec[offset + reg_b] = 1.0

        offset += self.num_registers
        # val (0..num_values-1)
        if 0 <= val < self.num_values:
            vec[offset + val] = 1.0

        return vec

    def generate_single_sequence(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates a single procedural episode.

        Returns:
            inputs: (sequence_length, input_dim)
            step_targets: (sequence_length, num_registers)
            final_target: (num_registers,)
        """
        # All registers initialized to 0
        registers = np.zeros(self.num_registers, dtype=np.int64)

        inputs_list: List[np.ndarray] = []
        step_targets_list: List[np.ndarray] = []

        for step in range(self.sequence_length):
            # To ensure non-trivial states early on, bias the first 2 steps toward SET operations
            if step < 2:
                op_type = self.OP_SET
            else:
                if self.op_weights is not None:
                    op_type = int(rng.choice(self.NUM_OPS, p=self.op_weights))
                else:
                    op_type = int(rng.integers(0, self.NUM_OPS))

            if op_type == self.OP_SET:
                reg_a = int(rng.integers(0, self.num_registers))
                reg_b = -1
                val = int(rng.integers(0, self.num_values))
                registers[reg_a] = val

            elif op_type == self.OP_SWAP:
                reg_a = int(rng.integers(0, self.num_registers))
                # Choose reg_b != reg_a
                candidates = [r for r in range(self.num_registers) if r != reg_a]
                reg_b = int(rng.choice(candidates))
                val = -1
                registers[reg_a], registers[reg_b] = registers[reg_b], registers[reg_a]

            elif op_type == self.OP_ADD:
                reg_a = int(rng.integers(0, self.num_registers))
                reg_b = int(rng.integers(0, self.num_registers))
                val = -1
                registers[reg_a] = (registers[reg_a] + registers[reg_b]) % self.num_values
            else:
                raise ValueError(f"Unknown op_type: {op_type}")

            vec = self._encode_operation(op_type, reg_a, reg_b, val)
            inputs_list.append(vec)
            step_targets_list.append(registers.copy())

        inputs_arr = np.stack(inputs_list, axis=0)  # (T, input_dim)
        step_targets_arr = np.stack(step_targets_list, axis=0)  # (T, num_registers)
        final_target = registers.copy()  # (num_registers,)

        return inputs_arr, step_targets_arr, final_target

    def generate_batch(self, batch_size: int, seed: Optional[int] = None) -> TaskBatch:
        """Generates a batch of procedural Register Tracking sequences.

        Args:
            batch_size: Number of sequences in the batch.
            seed: Optional seed for deterministic reproducibility.

        Returns:
            TaskBatch with:
                inputs: FloatTensor of shape (B, T, input_dim)
                targets: LongTensor of shape (B, num_registers)
                metadata: {"step_targets": LongTensor of shape (B, T, num_registers)}
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng

        batch_inputs: List[np.ndarray] = []
        batch_step_targets: List[np.ndarray] = []
        batch_final_targets: List[np.ndarray] = []

        for _ in range(batch_size):
            inp, step_tgt, final_tgt = self.generate_single_sequence(rng)
            batch_inputs.append(inp)
            batch_step_targets.append(step_tgt)
            batch_final_targets.append(final_tgt)

        inputs_t = torch.from_numpy(np.stack(batch_inputs, axis=0)).float()
        targets_t = torch.from_numpy(np.stack(batch_final_targets, axis=0)).long()
        step_targets_t = torch.from_numpy(np.stack(batch_step_targets, axis=0)).long()

        return TaskBatch(
            inputs=inputs_t,
            targets=targets_t,
            metadata={"step_targets": step_targets_t},
        )

    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Evaluates model predictions against ground truth targets.

        Args:
            predictions: Model output, either logits of shape (B, num_registers, num_values)
                         or predicted integer labels of shape (B, num_registers).
            targets: LongTensor of shape (B, num_registers).

        Returns:
            Dictionary with 'exact_match' and 'per_register_accuracy'.
        """
        if predictions.ndim == 3:
            pred_indices = predictions.argmax(dim=-1)
        elif predictions.ndim == 2:
            pred_indices = predictions
        else:
            raise ValueError(f"Expected predictions with 2 or 3 dims, got shape {predictions.shape}")

        exact_match = (pred_indices == targets).all(dim=-1).float().mean().item()
        per_reg_acc = (pred_indices == targets).float().mean().item()

        return {
            "exact_match": exact_match,
            "per_register_accuracy": per_reg_acc,
        }
