"""Synthetic deterministic algorithmic benchmark tasks for Phase 1.5 audit.

Provides 4 independent state-tracking tasks preserving the core abstraction
(4 integer registers in {0..9}, sequence length 10, input dimension 21,
terminal supervision at t=10):
1. MultiRegisterCopyTask: Copy, assignment, and broadcast.
2. PermutationCycleTask: Global cyclic shifts (left/right) and swaps.
3. RunningAccumulationTask: Running modular accumulation, addition, and zero reset.
4. ConditionalOverwriteTask: Value-comparison conditional updates.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch

from re_reason.benchmarks.base import BaseTask, TaskBatch


class BaseAlgorithmicTask(BaseTask):
    """Common foundation for 21-dimensional 4-register algorithmic tasks."""

    def __init__(
        self,
        num_registers: int = 4,
        sequence_length: int = 10,
        num_values: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        self.num_registers = num_registers
        self.sequence_length = sequence_length
        self.num_values = num_values
        self.input_dim = 3 + num_registers + num_registers + num_values  # 21 dims
        self._rng = np.random.default_rng(seed if seed is not None else 42)

    def _encode_step(self, op_type: int, reg_a: int, reg_b: int, val: int) -> np.ndarray:
        vec = np.zeros(self.input_dim, dtype=np.float32)
        vec[op_type] = 1.0
        offset = 3
        if 0 <= reg_a < self.num_registers:
            vec[offset + reg_a] = 1.0
        offset += self.num_registers
        if 0 <= reg_b < self.num_registers:
            vec[offset + reg_b] = 1.0
        offset += self.num_registers
        if 0 <= val < self.num_values:
            vec[offset + val] = 1.0
        return vec

    def generate_batch(self, batch_size: int, seed: Optional[int] = None) -> TaskBatch:
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        batch_inputs = []
        batch_step_targets = []
        batch_final_targets = []

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

    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor):
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


class MultiRegisterCopyTask(BaseAlgorithmicTask):
    """Task 1: Direct register copy, assignment, and global broadcast.

    Operations:
        0: COPY reg_a = reg_b
        1: SET reg_a = val
        2: BROADCAST all registers = reg_a
    """

    OP_COPY = 0
    OP_SET = 1
    OP_BROADCAST = 2

    def generate_single_sequence(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        registers = np.zeros(self.num_registers, dtype=np.int64)
        inputs_list = []
        step_targets_list = []

        for step in range(self.sequence_length):
            if step < 2:
                op_type = self.OP_SET
            else:
                op_type = int(rng.integers(0, 3))

            reg_a = int(rng.integers(0, self.num_registers))
            reg_b = int(rng.integers(0, self.num_registers))
            val = int(rng.integers(0, self.num_values))

            if op_type == self.OP_COPY:
                registers[reg_a] = registers[reg_b]
            elif op_type == self.OP_SET:
                registers[reg_a] = val
            elif op_type == self.OP_BROADCAST:
                registers[:] = registers[reg_a]

            vec = self._encode_step(op_type, reg_a, reg_b, val)
            inputs_list.append(vec)
            step_targets_list.append(registers.copy())

        return (
            np.stack(inputs_list, axis=0),
            np.stack(step_targets_list, axis=0),
            registers.copy(),
        )


class PermutationCycleTask(BaseAlgorithmicTask):
    """Task 2: Rigid topological register permutations and cyclic shifts.

    Operations:
        0: CYCLE_RIGHT (R0, R1, R2, R3) <- (R3, R0, R1, R2)
        1: CYCLE_LEFT  (R0, R1, R2, R3) <- (R1, R2, R3, R0)
        2: SWAP reg_a, reg_b
    """

    OP_CYCLE_RIGHT = 0
    OP_CYCLE_LEFT = 1
    OP_SWAP = 2

    def generate_single_sequence(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Initialize with distinct random values in first 2 steps
        registers = np.array([int(rng.integers(0, self.num_values)) for _ in range(self.num_registers)], dtype=np.int64)
        inputs_list = []
        step_targets_list = []

        for step in range(self.sequence_length):
            op_type = int(rng.integers(0, 3))
            reg_a = int(rng.integers(0, self.num_registers))
            reg_b = int(rng.integers(0, self.num_registers))
            val = int(rng.integers(0, self.num_values))

            if op_type == self.OP_CYCLE_RIGHT:
                registers = np.roll(registers, shift=1)
            elif op_type == self.OP_CYCLE_LEFT:
                registers = np.roll(registers, shift=-1)
            elif op_type == self.OP_SWAP:
                registers[reg_a], registers[reg_b] = registers[reg_b], registers[reg_a]

            vec = self._encode_step(op_type, reg_a, reg_b, val)
            inputs_list.append(vec)
            step_targets_list.append(registers.copy())

        return (
            np.stack(inputs_list, axis=0),
            np.stack(step_targets_list, axis=0),
            registers.copy(),
        )


class RunningAccumulationTask(BaseAlgorithmicTask):
    """Task 3: Running modular arithmetic accumulation and selective clearing.

    Operations:
        0: ACCUM reg_a = (reg_a + val) % 10
        1: ADD reg_a = (reg_a + reg_b) % 10
        2: RESET reg_a = 0
    """

    OP_ACCUM = 0
    OP_ADD = 1
    OP_RESET = 2

    def generate_single_sequence(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        registers = np.zeros(self.num_registers, dtype=np.int64)
        inputs_list = []
        step_targets_list = []

        for step in range(self.sequence_length):
            if step < 2:
                op_type = self.OP_ACCUM
            else:
                op_type = int(rng.integers(0, 3))

            reg_a = int(rng.integers(0, self.num_registers))
            reg_b = int(rng.integers(0, self.num_registers))
            val = int(rng.integers(0, self.num_values))

            if op_type == self.OP_ACCUM:
                registers[reg_a] = (registers[reg_a] + val) % self.num_values
            elif op_type == self.OP_ADD:
                registers[reg_a] = (registers[reg_a] + registers[reg_b]) % self.num_values
            elif op_type == self.OP_RESET:
                registers[reg_a] = 0

            vec = self._encode_step(op_type, reg_a, reg_b, val)
            inputs_list.append(vec)
            step_targets_list.append(registers.copy())

        return (
            np.stack(inputs_list, axis=0),
            np.stack(step_targets_list, axis=0),
            registers.copy(),
        )


class ConditionalOverwriteTask(BaseAlgorithmicTask):
    """Task 4: State-dependent conditional overwrite (branchless logic).

    Operations:
        0: COND_GE: if reg_a >= reg_b, reg_a = val
        1: COND_LT: if reg_a < reg_b, reg_a = val
        2: UNCOND: reg_a = val
    """

    OP_COND_GE = 0
    OP_COND_LT = 1
    OP_UNCOND = 2

    def generate_single_sequence(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        registers = np.zeros(self.num_registers, dtype=np.int64)
        inputs_list = []
        step_targets_list = []

        for step in range(self.sequence_length):
            if step < 2:
                op_type = self.OP_UNCOND
            else:
                op_type = int(rng.integers(0, 3))

            reg_a = int(rng.integers(0, self.num_registers))
            reg_b = int(rng.integers(0, self.num_registers))
            val = int(rng.integers(0, self.num_values))

            if op_type == self.OP_COND_GE:
                if registers[reg_a] >= registers[reg_b]:
                    registers[reg_a] = val
            elif op_type == self.OP_COND_LT:
                if registers[reg_a] < registers[reg_b]:
                    registers[reg_a] = val
            elif op_type == self.OP_UNCOND:
                registers[reg_a] = val

            vec = self._encode_step(op_type, reg_a, reg_b, val)
            inputs_list.append(vec)
            step_targets_list.append(registers.copy())

        return (
            np.stack(inputs_list, axis=0),
            np.stack(step_targets_list, axis=0),
            registers.copy(),
        )
