"""Phase 1.6 Procedural Benchmark Generators, Split Managers, and Data Contracts.

Implements the authoritative Phase 1.6 data contracts:
1. 22-dimensional instruction encoding (SET, SWAP, READ, ADD).
2. Canonical 3-step Stage 1 primitive trace generator (SET r0 a, SET r1 b, ADD r0 r1).
3. Stage 1 closed-form modular pair partition (64 train / 16 val / 20 held-out).
4. Stage 2 pure address-value binding sequences (SET, SWAP, READ, zero ADD).
5. Stage 3 composed in-distribution sequences (Regime A).
6. Stage 4 systematic generalisation sequences with execution-time source operand
   masking (Regime B: runtime v(r_src) not in {0, 5} on ADD) and conditioned
   held-out test sequences (requiring at least one ADD with runtime v(r_src) in {0, 5}).
7. Independent deterministic replay simulator for trace verification.
8. Cryptographically verified sequence-disjoint split managers.
"""

from collections import Counter
import hashlib
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple
import numpy as np
import torch

from re_reason.benchmarks.base import BaseTask, TaskBatch
from re_reason.benchmarks.dataset_split import (
    SplitDataset,
    assert_manifests_disjoint,
    build_hash_manifest,
    hash_sequence,
)


class Instruction(NamedTuple):
    """Decoded instruction representation for verification and replay."""
    op_type: int
    reg_dest: int
    reg_src: int
    immediate_val: int


class Phase16RegisterTask(BaseTask):
    """Phase 1.6 Register Tracking Task.

    Registers: 4 integer registers (R0, R1, R2, R3) taking values in {0, ..., 9}.
    Default Horizon: sequence_length = 10 (or 3 for Stage 1 primitive traces).

    Instruction Encoding (22 dimensions):
        e_op: 4-dim one-hot operation type
            0: SET  (r_dest, no_src, val)    -> R[r_dest] = val
            1: SWAP (r_dest, r_src, no_val)  -> R[r_dest], R[r_src] = R[r_src], R[r_dest]
            2: READ (r_dest, no_src, no_val) -> State unmutated, queries R[r_dest]
            3: ADD  (r_dest, r_src, no_val)  -> R[r_dest] = (R[r_dest] + R[r_src]) % 10
        e(r_dest): 4-dim one-hot destination register address (indices 4..7)
        e(r_src):  4-dim one-hot source register address (indices 8..11),
                   strictly zero vector 0_4 for SET and READ (no-source convention).
        e(v):      10-dim one-hot immediate value in {0..9} (indices 12..21),
                   strictly zero vector 0_10 for SWAP, READ, and ADD.
    """

    OP_SET = 0
    OP_SWAP = 1
    OP_READ = 2
    OP_ADD = 3
    NUM_OPS = 4

    NUM_REGISTERS = 4
    NUM_VALUES = 10
    INPUT_DIM = 4 + 4 + 4 + 10  # 22 dimensions

    def __init__(
        self,
        num_registers: int = 4,
        sequence_length: int = 10,
        num_values: int = 10,
        seed: Optional[int] = None,
        op_weights: Optional[Tuple[float, float, float, float]] = None,
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

    def encode_instruction(
        self,
        op_type: int,
        reg_dest: int,
        reg_src: int = -1,
        immediate_val: int = -1,
    ) -> np.ndarray:
        """Encodes an operation into the 22-dimensional instruction token.

        Enforces:
        - SET: reg_dest in [0..3], reg_src = -1 (encodes 0_4), immediate_val in [0..9].
        - SWAP: reg_dest in [0..3], reg_src in [0..3], immediate_val = -1 (encodes 0_10).
        - READ: reg_dest in [0..3], reg_src = -1 (encodes 0_4), immediate_val = -1 (encodes 0_10).
        - ADD: reg_dest in [0..3], reg_src in [0..3], immediate_val = -1 (encodes 0_10).
        """
        vec = np.zeros(self.input_dim, dtype=np.float32)

        # 1. Operation type one-hot (indices 0..3)
        if not (0 <= op_type < self.NUM_OPS):
            raise ValueError(f"Invalid op_type {op_type}, must be in [0, {self.NUM_OPS - 1}]")
        vec[op_type] = 1.0

        # 2. Destination register address one-hot (indices 4..7)
        offset = self.NUM_OPS
        if not (0 <= reg_dest < self.num_registers):
            raise ValueError(f"Invalid reg_dest {reg_dest}, must be in [0, {self.num_registers - 1}]")
        vec[offset + reg_dest] = 1.0

        # 3. Source register address one-hot (indices 8..11)
        offset += self.num_registers
        if op_type in (self.OP_SWAP, self.OP_ADD):
            if not (0 <= reg_src < self.num_registers):
                raise ValueError(f"Operation {op_type} requires valid reg_src in [0, {self.num_registers - 1}], got {reg_src}")
            vec[offset + reg_src] = 1.0
        else:
            # Documented no-source convention: strictly 0_4 for SET and READ
            if reg_src != -1 and reg_src is not None:
                raise ValueError(f"Operation {op_type} must not specify reg_src (expected -1, got {reg_src})")

        # 4. Immediate literal value one-hot (indices 12..21)
        offset += self.num_registers
        if op_type == self.OP_SET:
            if not (0 <= immediate_val < self.num_values):
                raise ValueError(f"SET requires valid immediate_val in [0, {self.num_values - 1}], got {immediate_val}")
            vec[offset + immediate_val] = 1.0
        else:
            # Strict no-leakage invariant: zero vector for SWAP, READ, ADD
            if immediate_val != -1 and immediate_val is not None:
                raise ValueError(f"Operation {op_type} must not specify immediate_val (expected -1, got {immediate_val})")

        return vec

    def decode_instruction(self, token: np.ndarray) -> Instruction:
        """Decodes a 22-dimensional instruction token back to an Instruction namedtuple."""
        if token.shape[0] != self.input_dim:
            raise ValueError(f"Expected token of length {self.input_dim}, got {token.shape[0]}")

        if not np.all((token == 0.0) | (token == 1.0)):
            raise ValueError("Token elements must be strictly binary in {0.0, 1.0}")

        op_slice = token[:self.NUM_OPS]
        if np.count_nonzero(op_slice == 1.0) != 1 or op_slice.sum() != 1.0:
            raise ValueError("Token must have exactly one active op_type")
        op_type = int(np.argmax(op_slice))

        dest_slice = token[self.NUM_OPS : self.NUM_OPS + self.num_registers]
        if np.count_nonzero(dest_slice == 1.0) != 1 or dest_slice.sum() != 1.0:
            raise ValueError("Token must have exactly one active destination register")
        reg_dest = int(np.argmax(dest_slice))

        src_offset = self.NUM_OPS + self.num_registers
        src_slice = token[src_offset : src_offset + self.num_registers]
        if op_type in (self.OP_SWAP, self.OP_ADD):
            if np.count_nonzero(src_slice == 1.0) != 1 or src_slice.sum() != 1.0:
                raise ValueError(f"Op {op_type} requires exactly one active source register")
            reg_src = int(np.argmax(src_slice))
        else:
            if np.count_nonzero(src_slice == 0.0) != self.num_registers or src_slice.sum() != 0.0:
                raise ValueError(f"Op {op_type} must have zero source register slice")
            reg_src = -1

        val_offset = src_offset + self.num_registers
        val_slice = token[val_offset : val_offset + self.num_values]
        if op_type == self.OP_SET:
            if np.count_nonzero(val_slice == 1.0) != 1 or val_slice.sum() != 1.0:
                raise ValueError("SET requires exactly one active immediate value")
            immediate_val = int(np.argmax(val_slice))
        else:
            if np.count_nonzero(val_slice == 0.0) != self.num_values or val_slice.sum() != 0.0:
                raise ValueError(f"Op {op_type} must have strictly zero immediate value slice (leakage detected)")
            immediate_val = -1

        return Instruction(op_type, reg_dest, reg_src, immediate_val)

    def execute_step(
        self,
        registers: np.ndarray,
        instruction: Instruction,
    ) -> Tuple[np.ndarray, Optional[int]]:
        """Executes a single instruction transition on the register state.

        Args:
            registers: Current state of shape (num_registers,).
            instruction: Decoded instruction to execute.

        Returns:
            Tuple of (new_registers_copy, read_output_if_applicable).
        """
        new_regs = registers.copy()
        read_val = None

        if instruction.op_type == self.OP_SET:
            new_regs[instruction.reg_dest] = instruction.immediate_val
        elif instruction.op_type == self.OP_SWAP:
            d, s = instruction.reg_dest, instruction.reg_src
            new_regs[d], new_regs[s] = new_regs[s], new_regs[d]
        elif instruction.op_type == self.OP_READ:
            # READ causes zero mutation across all registers
            read_val = int(new_regs[instruction.reg_dest])
        elif instruction.op_type == self.OP_ADD:
            d, s = instruction.reg_dest, instruction.reg_src
            new_regs[d] = (new_regs[d] + new_regs[s]) % self.num_values
        else:
            raise ValueError(f"Unknown op_type: {instruction.op_type}")

        return new_regs, read_val

    def simulate_trace(
        self,
        inputs_arr: np.ndarray,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Replays an encoded sequence of instructions from ground state 0_4.

        Validates all transitions, verifies exact post-step states, and records
        per-step execution metadata (including runtime operand values).

        Returns:
            step_targets: Array of shape (T, num_registers).
            trace_history: List of step dictionaries recording runtime execution values.
        """
        t_len = inputs_arr.shape[0]
        registers = np.zeros(self.num_registers, dtype=np.int64)
        step_targets = np.zeros((t_len, self.num_registers), dtype=np.int64)
        history: List[Dict[str, Any]] = []

        for t in range(t_len):
            inst = self.decode_instruction(inputs_arr[t])
            # Record runtime pre-execution operand values
            dest_pre = int(registers[inst.reg_dest])
            src_pre = int(registers[inst.reg_src]) if inst.reg_src != -1 else None

            registers, read_val = self.execute_step(registers, inst)
            step_targets[t] = registers.copy()

            history.append({
                "step": t,
                "op_type": inst.op_type,
                "reg_dest": inst.reg_dest,
                "reg_src": inst.reg_src,
                "immediate_val": inst.immediate_val,
                "runtime_dest_val_pre": dest_pre,
                "runtime_src_val_pre": src_pre,
                "runtime_dest_val_post": int(registers[inst.reg_dest]),
                "read_output": read_val,
            })

        return step_targets, history

    def generate_stage1_trace(
        self,
        a: int,
        b: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates the canonical 3-step Stage 1 primitive trace for operand pair (a, b).

        Trace:
            t=1 (step 0): SET r0 a
            t=2 (step 1): SET r1 b
            t=3 (step 2): ADD r0 r1

        Returns:
            inputs: Array of shape (3, 22).
            step_targets: Array of shape (3, 4).
            final_target: Array of shape (4,), with final_target[0] == (a + b) % 10.
        """
        if not (0 <= a < self.num_values and 0 <= b < self.num_values):
            raise ValueError(f"Operands must be in [0, {self.num_values - 1}], got a={a}, b={b}")

        inputs = np.zeros((3, self.input_dim), dtype=np.float32)
        # Step 1: SET r0 a
        inputs[0] = self.encode_instruction(self.OP_SET, reg_dest=0, reg_src=-1, immediate_val=a)
        # Step 2: SET r1 b
        inputs[1] = self.encode_instruction(self.OP_SET, reg_dest=1, reg_src=-1, immediate_val=b)
        # Step 3: ADD r0 r1
        inputs[2] = self.encode_instruction(self.OP_ADD, reg_dest=0, reg_src=1, immediate_val=-1)

        step_targets, _ = self.simulate_trace(inputs)
        final_target = step_targets[-1].copy()

        expected_sum = (a + b) % self.num_values
        if final_target[0] != expected_sum:
            raise RuntimeError(f"Arithmetic fault in Stage 1 trace: expected {expected_sum}, got {final_target[0]}")

        return inputs, step_targets, final_target

    def generate_single_sequence(
        self,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates a standard unmasked composed sequence (Regime A / in-distribution).

        Returns:
            inputs: Array of shape (sequence_length, 22).
            step_targets: Array of shape (sequence_length, 4).
            final_target: Array of shape (4,).
        """
        inputs_list: List[np.ndarray] = []
        registers = np.zeros(self.num_registers, dtype=np.int64)

        for step in range(self.sequence_length):
            # Establish non-trivial state early
            if step < 2:
                op_type = self.OP_SET
            else:
                if self.op_weights is not None:
                    op_type = int(rng.choice(self.NUM_OPS, p=self.op_weights))
                else:
                    op_type = int(rng.integers(0, self.NUM_OPS))

            if op_type == self.OP_SET:
                reg_dest = int(rng.integers(0, self.num_registers))
                reg_src = -1
                val = int(rng.integers(0, self.num_values))
            elif op_type == self.OP_SWAP:
                reg_dest = int(rng.integers(0, self.num_registers))
                candidates = [r for r in range(self.num_registers) if r != reg_dest]
                reg_src = int(rng.choice(candidates)) if candidates else reg_dest
                val = -1
            elif op_type == self.OP_READ:
                reg_dest = int(rng.integers(0, self.num_registers))
                reg_src = -1
                val = -1
            elif op_type == self.OP_ADD:
                reg_dest = int(rng.integers(0, self.num_registers))
                reg_src = int(rng.integers(0, self.num_registers))
                val = -1
            else:
                raise ValueError(f"Unknown op_type: {op_type}")

            token = self.encode_instruction(op_type, reg_dest, reg_src, val)
            inputs_list.append(token)
            registers, _ = self.execute_step(registers, Instruction(op_type, reg_dest, reg_src, val))

        inputs_arr = np.stack(inputs_list, axis=0)
        step_targets, _ = self.simulate_trace(inputs_arr)
        final_target = step_targets[-1].copy()

        return inputs_arr, step_targets, final_target

    def generate_stage2_binding_sequence(
        self,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates a Stage 2 pure address-value binding sequence.

        Format: Multi-step sequence (T=10) consisting strictly of SET, SWAP, and READ.
        Enforces: Exactly zero ADD operations throughout the sequence.
        """
        inputs_list: List[np.ndarray] = []
        registers = np.zeros(self.num_registers, dtype=np.int64)
        allowed_ops = [self.OP_SET, self.OP_SWAP, self.OP_READ]

        for step in range(self.sequence_length):
            if step < 2:
                op_type = self.OP_SET
            elif step == self.sequence_length - 1:
                # Terminal step queries a register via READ
                op_type = self.OP_READ
            else:
                op_type = int(rng.choice(allowed_ops))

            if op_type == self.OP_SET:
                reg_dest = int(rng.integers(0, self.num_registers))
                reg_src = -1
                val = int(rng.integers(0, self.num_values))
            elif op_type == self.OP_SWAP:
                reg_dest = int(rng.integers(0, self.num_registers))
                candidates = [r for r in range(self.num_registers) if r != reg_dest]
                reg_src = int(rng.choice(candidates)) if candidates else reg_dest
                val = -1
            elif op_type == self.OP_READ:
                reg_dest = int(rng.integers(0, self.num_registers))
                reg_src = -1
                val = -1
            else:
                raise RuntimeError(f"ADD operation leaked into Stage 2 generator: {op_type}")

            token = self.encode_instruction(op_type, reg_dest, reg_src, val)
            inputs_list.append(token)
            registers, _ = self.execute_step(registers, Instruction(op_type, reg_dest, reg_src, val))

        inputs_arr = np.stack(inputs_list, axis=0)
        step_targets, history = self.simulate_trace(inputs_arr)

        # Verification assertion: zero ADD transitions
        assert all(h["op_type"] != self.OP_ADD for h in history), "ADD leaked into Stage 2 sequence"

        return inputs_arr, step_targets, step_targets[-1].copy()

    def generate_regime_b_sequence(
        self,
        rng: np.random.Generator,
        max_attempts: int = 100,
        excluded_source_values: Tuple[int, ...] = (0, 5),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates a Regime B training/validation sequence with execution-time operand masking.

        Invariant: Every ADD transition in the sequence must satisfy:
            runtime source-register value v(r_src) not in ``excluded_source_values``
            (default {0, 5}; the 2026-10-18 held-out campaign registers {0, 3, 5}).
        Immediate literals in SET instructions remain unrestricted (0..9); masking
        applies strictly to the runtime source value of ADD transitions. The check is
        evaluated at the exact runtime moment of the transition, accounting for all
        preceding SET, SWAP, and ADD mutations.
        """
        if not excluded_source_values:
            raise ValueError("excluded_source_values must be non-empty")
        if any(v not in range(self.num_values) for v in excluded_source_values):
            raise ValueError(
                f"excluded_source_values must be within [0, {self.num_values - 1}], "
                f"got {excluded_source_values!r}"
            )
        excluded = tuple(sorted(set(int(v) for v in excluded_source_values)))
        if len(set(range(self.num_values)) - set(excluded)) == 0:
            raise ValueError(
                "excluded_source_values must leave at least one eligible runtime source value"
            )

        for _ in range(max_attempts):
            inputs_list: List[np.ndarray] = []
            registers = np.zeros(self.num_registers, dtype=np.int64)
            valid_sequence = True

            for step in range(self.sequence_length):
                if step < 2:
                    op_type = self.OP_SET
                else:
                    op_type = int(rng.integers(0, self.NUM_OPS))

                if op_type == self.OP_SET:
                    reg_dest = int(rng.integers(0, self.num_registers))
                    reg_src = -1
                    val = int(rng.integers(0, self.num_values))

                elif op_type == self.OP_SWAP:
                    reg_dest = int(rng.integers(0, self.num_registers))
                    candidates = [r for r in range(self.num_registers) if r != reg_dest]
                    reg_src = int(rng.choice(candidates)) if candidates else reg_dest
                    val = -1

                elif op_type == self.OP_READ:
                    reg_dest = int(rng.integers(0, self.num_registers))
                    reg_src = -1
                    val = -1

                elif op_type == self.OP_ADD:
                    reg_dest = int(rng.integers(0, self.num_registers))
                    # Pick reg_src such that registers[reg_src] not in the excluded set
                    eligible_src = [r for r in range(self.num_registers) if registers[r] not in excluded]
                    if not eligible_src:
                        # No register currently holds an eligible operand; fallback to SET or resample
                        val = int(rng.choice([v for v in range(self.num_values) if v not in excluded]))
                        reg_dest = int(rng.integers(0, self.num_registers))
                        op_type = self.OP_SET
                        reg_src = -1
                    else:
                        reg_src = int(rng.choice(eligible_src))
                        val = -1
                        # Runtime invariant verification
                        if registers[reg_src] in excluded:
                            valid_sequence = False
                            break
                else:
                    raise ValueError(f"Unknown op_type: {op_type}")

                token = self.encode_instruction(op_type, reg_dest, reg_src, val)
                inputs_list.append(token)
                registers, _ = self.execute_step(registers, Instruction(op_type, reg_dest, reg_src, val))

            if not valid_sequence:
                continue

            inputs_arr = np.stack(inputs_list, axis=0)
            step_targets, history = self.simulate_trace(inputs_arr)

            # Replay verification: verify 100% of ADD steps exclude the masked runtime values
            add_steps = [h for h in history if h["op_type"] == self.OP_ADD]
            violating_add = [h for h in add_steps if h["runtime_src_val_pre"] in excluded]
            if not violating_add:
                return inputs_arr, step_targets, step_targets[-1].copy()

        raise RuntimeError(f"Failed to generate valid Regime B masked sequence after {max_attempts} attempts")

    def generate_proxy_dev_sequence(
        self,
        rng: np.random.Generator,
        target_class: int,
        max_attempts: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates one PROXY-DEV development sequence for a requested target class.

        Conditioning invariants (registered 2026-10-18 Phase 1.6 amendment):
        * exactly one designated ADD transition per sequence with runtime source value 3;
        * zero ADD transitions with runtime source value in {0, 5};
        * any other ADD transitions must have runtime sources in {1, 2, 4, 6, 7, 8, 9};
        * the designated ADD's post-`ADD` target class ``(v(r_dest) + 3) mod 10``
          equals ``target_class``.

        Randomized trace details: designated ADD step position, destination/source
        register addresses, intervening operations, and SET literals (0..9 remain
        unrestricted). Every accepted sequence passes deterministic replay verification
        of all invariants. Attempt exhaustion raises a fatal ``RuntimeError``; the
        sample count is never silently reduced and the mask is never relaxed.
        """
        if self.sequence_length < 3:
            raise ValueError("PROXY-DEV generation requires sequence_length >= 3")
        if not (0 <= target_class < self.num_values):
            raise ValueError(f"target_class must be in [0, {self.num_values - 1}], got {target_class}")

        designated_src_value = 3
        dest_literal = (target_class - designated_src_value) % self.num_values
        excluded = (0, 3, 5)
        seen_only = (1, 2, 4, 6, 7, 8, 9)

        for _ in range(max_attempts):
            inputs_list: List[np.ndarray] = []
            registers = np.zeros(self.num_registers, dtype=np.int64)

            designated_step = int(rng.integers(2, self.sequence_length))
            src_reg = int(rng.integers(0, self.num_registers))
            dest_candidates = [r for r in range(self.num_registers) if r != src_reg]
            dest_reg = int(rng.choice(dest_candidates)) if dest_candidates else src_reg

            for step in range(self.sequence_length):
                if step == designated_step - 2:
                    op_type, reg_dest, reg_src, val = self.OP_SET, src_reg, -1, designated_src_value
                elif step == designated_step - 1:
                    op_type, reg_dest, reg_src, val = self.OP_SET, dest_reg, -1, dest_literal
                elif step == designated_step:
                    op_type, reg_dest, reg_src, val = self.OP_ADD, dest_reg, src_reg, -1
                else:
                    op_type = int(rng.integers(0, self.NUM_OPS))
                    if op_type == self.OP_SET:
                        reg_dest = int(rng.integers(0, self.num_registers))
                        reg_src = -1
                        val = int(rng.integers(0, self.num_values))
                    elif op_type == self.OP_SWAP:
                        reg_dest = int(rng.integers(0, self.num_registers))
                        candidates = [r for r in range(self.num_registers) if r != reg_dest]
                        reg_src = int(rng.choice(candidates)) if candidates else reg_dest
                        val = -1
                    elif op_type == self.OP_READ:
                        reg_dest = int(rng.integers(0, self.num_registers))
                        reg_src = -1
                        val = -1
                    elif op_type == self.OP_ADD:
                        reg_dest = int(rng.integers(0, self.num_registers))
                        # Non-designated ADDs must source the seen set only.
                        eligible_src = [
                            r for r in range(self.num_registers) if registers[r] not in excluded
                        ]
                        if not eligible_src:
                            val = int(rng.choice(seen_only))
                            reg_dest = int(rng.integers(0, self.num_registers))
                            op_type = self.OP_SET
                            reg_src = -1
                        else:
                            reg_src = int(rng.choice(eligible_src))
                            val = -1
                    else:
                        raise ValueError(f"Unknown op_type: {op_type}")

                token = self.encode_instruction(op_type, reg_dest, reg_src, val)
                inputs_list.append(token)
                registers, _ = self.execute_step(registers, Instruction(op_type, reg_dest, reg_src, val))

            inputs_arr = np.stack(inputs_list, axis=0)
            step_targets, history = self.simulate_trace(inputs_arr)

            add_steps = [h for h in history if h["op_type"] == self.OP_ADD]
            designated = [h for h in add_steps if h["runtime_src_val_pre"] == designated_src_value]
            forbidden_05 = [h for h in add_steps if h["runtime_src_val_pre"] in (0, 5)]
            outside_seen = [
                h
                for h in add_steps
                if h["runtime_src_val_pre"] not in seen_only
                and h["runtime_src_val_pre"] != designated_src_value
            ]

            class_ok = (
                len(designated) == 1
                and (designated[0]["runtime_dest_val_pre"] + designated_src_value) % self.num_values
                == target_class
            )
            if class_ok and not forbidden_05 and not outside_seen:
                return inputs_arr, step_targets, step_targets[-1].copy()

        raise RuntimeError(
            f"Failed to generate PROXY-DEV sequence for target class {target_class} "
            f"after {max_attempts} attempts"
        )

    def generate_regime_b_heldout_test_sequence(
        self,
        rng: np.random.Generator,
        max_attempts: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates a Regime B held-out test sequence.

        Conditioning Invariant: The sequence MUST contain at least one ADD transition
        where the runtime source-register value v(r_src) in {0, 5} at execution time.
        """
        for _ in range(max_attempts):
            inputs_list: List[np.ndarray] = []
            registers = np.zeros(self.num_registers, dtype=np.int64)

            # Ensure an eligible value (0 or 5) is placed into a register early
            target_step_for_heldout = int(rng.integers(2, self.sequence_length))
            heldout_src_reg = int(rng.integers(0, self.num_registers))

            for step in range(self.sequence_length):
                if step == 0:
                    # Place 0 or 5 into heldout_src_reg at step 0
                    op_type = self.OP_SET
                    reg_dest = heldout_src_reg
                    reg_src = -1
                    val = int(rng.choice([0, 5]))

                elif step == target_step_for_heldout:
                    # Execute ADD sourcing heldout_src_reg if it still holds {0, 5}
                    op_type = self.OP_ADD
                    candidates_dest = [r for r in range(self.num_registers) if r != heldout_src_reg]
                    reg_dest = int(rng.choice(candidates_dest)) if candidates_dest else 0
                    reg_src = heldout_src_reg
                    val = -1

                else:
                    op_type = int(rng.integers(0, self.NUM_OPS))
                    if op_type == self.OP_SET:
                        # Avoid overwriting heldout_src_reg before target step
                        if step < target_step_for_heldout:
                            candidates_dest = [r for r in range(self.num_registers) if r != heldout_src_reg]
                            reg_dest = int(rng.choice(candidates_dest)) if candidates_dest else 0
                        else:
                            reg_dest = int(rng.integers(0, self.num_registers))
                        reg_src = -1
                        val = int(rng.integers(0, self.num_values))
                    elif op_type == self.OP_SWAP:
                        reg_dest = int(rng.integers(0, self.num_registers))
                        candidates = [r for r in range(self.num_registers) if r != reg_dest]
                        reg_src = int(rng.choice(candidates)) if candidates else reg_dest
                        val = -1
                        if step < target_step_for_heldout:
                            # Track address shift if heldout_src_reg is swapped
                            if reg_dest == heldout_src_reg:
                                heldout_src_reg = reg_src
                            elif reg_src == heldout_src_reg:
                                heldout_src_reg = reg_dest
                    elif op_type == self.OP_READ:
                        reg_dest = int(rng.integers(0, self.num_registers))
                        reg_src = -1
                        val = -1
                    elif op_type == self.OP_ADD:
                        reg_dest = int(rng.integers(0, self.num_registers))
                        reg_src = int(rng.integers(0, self.num_registers))
                        val = -1
                    else:
                        raise ValueError(f"Unknown op_type: {op_type}")

                token = self.encode_instruction(op_type, reg_dest, reg_src, val)
                inputs_list.append(token)
                registers, _ = self.execute_step(registers, Instruction(op_type, reg_dest, reg_src, val))

            inputs_arr = np.stack(inputs_list, axis=0)
            step_targets, history = self.simulate_trace(inputs_arr)

            # Replay verification: verify at least one ADD step with runtime v(r_src) in {0, 5}
            add_heldout = [
                h for h in history
                if h["op_type"] == self.OP_ADD and h["runtime_src_val_pre"] in (0, 5)
            ]
            if len(add_heldout) >= 1:
                return inputs_arr, step_targets, step_targets[-1].copy()

        raise RuntimeError(
            f"Failed to generate conditioned Regime B held-out test sequence after {max_attempts} attempts"
        )

    def generate_batch(
        self,
        batch_size: int,
        seed: Optional[int] = None,
    ) -> TaskBatch:
        """Generates a batch of standard in-distribution (Regime A) sequences."""
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

    def evaluate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluates predictions against targets."""
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


class Stage1PairPartition:
    """Manages the deterministic 64/16/20 modular-pair partition for Stage 1.

    Mathematical Definition (Section 5.1 of proposal):
    - All 100 ordered operand pairs (a, b) in {0..9}^2.
    - Held-out Test (TEST-HELDOUT, 20 pairs): strictly all pairs with b in {0, 5}.
    - Seen Domain (80 pairs): all pairs with b in {1, 2, 3, 4, 6, 7, 8, 9}.
    - Validation (VAL-SEEN, 16 pairs): For column index i in 0..7 with sorted eligible
      b = [1, 2, 3, 4, 6, 7, 8, 9]:
          y0 = (2 * i) % 10,  y1 = (2 * i + 1) % 10
          a0 = (y0 - b) % 10, a1 = (y1 - b) % 10
      The pairs (a0, b) and (a1, b) form VAL-SEEN.
      Validation class frequencies: classes 0..5 occur twice, classes 6..9 occur once.
    - Training Fit (TRAIN-FIT, 64 pairs): remaining 8 values of a per column b.
    """

    ELIGIBLE_B = [1, 2, 3, 4, 6, 7, 8, 9]
    HELDOUT_B = [0, 5]

    def __init__(self) -> None:
        self.heldout_pairs: List[Tuple[int, int]] = sorted([
            (a, b) for b in self.HELDOUT_B for a in range(10)
        ])
        self.seen_pairs: List[Tuple[int, int]] = sorted([
            (a, b) for b in self.ELIGIBLE_B for a in range(10)
        ])

        val_set: Set[Tuple[int, int]] = set()
        val_targets_list: List[int] = []

        for i, b in enumerate(self.ELIGIBLE_B):
            y0 = (2 * i) % 10
            y1 = (2 * i + 1) % 10
            a0 = (y0 - b) % 10
            a1 = (y1 - b) % 10
            val_set.add((a0, b))
            val_set.add((a1, b))
            val_targets_list.extend([y0, y1])

        self.val_pairs: List[Tuple[int, int]] = sorted(list(val_set))
        self.val_targets: List[int] = val_targets_list
        self.train_pairs: List[Tuple[int, int]] = sorted(list(set(self.seen_pairs) - val_set))

        # Self-verification assertions
        assert len(self.heldout_pairs) == 20, f"Expected 20 heldout pairs, got {len(self.heldout_pairs)}"
        assert len(self.val_pairs) == 16, f"Expected 16 val pairs, got {len(self.val_pairs)}"
        assert len(self.train_pairs) == 64, f"Expected 64 train pairs, got {len(self.train_pairs)}"

        h_set = set(self.heldout_pairs)
        v_set = set(self.val_pairs)
        t_set = set(self.train_pairs)
        assert len(h_set & v_set) == 0, "Overlap between heldout and val pairs"
        assert len(h_set & t_set) == 0, "Overlap between heldout and train pairs"
        assert len(v_set & t_set) == 0, "Overlap between val and train pairs"
        assert len(h_set | v_set | t_set) == 100, "Partition does not cover all 100 pairs"

        counts = Counter(self.val_targets)
        expected_counts = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 1}
        assert counts == expected_counts, f"Validation class frequencies mismatch: {counts} vs {expected_counts}"

    def build_trace_split(
        self,
        pairs: List[Tuple[int, int]],
        task: Optional[Phase16RegisterTask] = None,
    ) -> SplitDataset:
        """Converts a list of operand pairs into a SplitDataset of canonical 3-step traces."""
        t = task if task is not None else Phase16RegisterTask(sequence_length=3)
        inputs_list: List[np.ndarray] = []
        step_targets_list: List[np.ndarray] = []
        final_targets_list: List[np.ndarray] = []
        hashes: List[str] = []

        for a, b in pairs:
            inp, step_tgt, final_tgt = t.generate_stage1_trace(a, b)
            h = hash_sequence(inp)
            hashes.append(h)
            inputs_list.append(inp)
            step_targets_list.append(step_tgt)
            final_targets_list.append(final_tgt)

        inputs_t = torch.from_numpy(np.stack(inputs_list, axis=0)).float()
        step_targets_t = torch.from_numpy(np.stack(step_targets_list, axis=0)).long()
        final_targets_t = torch.from_numpy(np.stack(final_targets_list, axis=0)).long()

        return SplitDataset(inputs_t, final_targets_t, step_targets_t, hashes)

    def get_stage1_splits(
        self,
        task: Optional[Phase16RegisterTask] = None,
    ) -> Tuple[SplitDataset, SplitDataset, SplitDataset]:
        """Returns the canonical (TRAIN-FIT, VAL-SEEN, TEST-HELDOUT) 3-step trace splits."""
        train_ds = self.build_trace_split(self.train_pairs, task)
        val_ds = self.build_trace_split(self.val_pairs, task)
        heldout_ds = self.build_trace_split(self.heldout_pairs, task)
        return train_ds, val_ds, heldout_ds


class Phase16SplitManager:
    """Manages strictly partitioned, sequence-disjoint, and verified unique datasets for Phase 1.6."""

    def __init__(self, task: Optional[Phase16RegisterTask] = None) -> None:
        self.task = task if task is not None else Phase16RegisterTask()

    def generate_disjoint_split(
        self,
        size: int,
        seed: int,
        generation_mode: str = "regime_a",
        max_attempts_factor: int = 20,
        excluded_source_values: Tuple[int, ...] = (0, 5),
    ) -> Tuple[SplitDataset, List[str]]:
        """Generates a verified sequence-disjoint split under the specified mode.

        Modes:
            - 'regime_a': Standard in-distribution unmasked composed sequences.
            - 'stage2_binding': Pure address binding sequences (SET, SWAP, READ, zero ADD).
            - 'regime_b_masked': Masked sequences excluding the configured runtime source
              operand values (default {0, 5}; the held-out campaign registers {0, 3, 5}).
            - 'regime_b_heldout_test': Conditioned sequences with >= 1 ADD sourcing runtime b in {0, 5}.
        """
        if generation_mode != "regime_b_masked" and tuple(excluded_source_values) != (0, 5):
            raise ValueError(
                "excluded_source_values are only supported for 'regime_b_masked' generation; "
                f"got mode {generation_mode!r}"
            )
        rng = np.random.default_rng(seed)
        seen_hashes: Set[str] = set()
        hashes_list: List[str] = []

        inputs_list: List[np.ndarray] = []
        step_targets_list: List[np.ndarray] = []
        final_targets_list: List[np.ndarray] = []

        total_attempts = 0
        max_attempts = size * max_attempts_factor

        while len(inputs_list) < size:
            total_attempts += 1
            if total_attempts > max_attempts:
                raise RuntimeError(
                    f"Exceeded max attempts ({max_attempts}) while generating {size} unique sequences "
                    f"under mode '{generation_mode}' (achieved {len(inputs_list)})."
                )

            if generation_mode == "regime_a":
                inp, step_tgt, final_tgt = self.task.generate_single_sequence(rng)
            elif generation_mode == "stage2_binding":
                inp, step_tgt, final_tgt = self.task.generate_stage2_binding_sequence(rng)
            elif generation_mode == "regime_b_masked":
                inp, step_tgt, final_tgt = self.task.generate_regime_b_sequence(
                    rng, excluded_source_values=excluded_source_values
                )
            elif generation_mode == "regime_b_heldout_test":
                inp, step_tgt, final_tgt = self.task.generate_regime_b_heldout_test_sequence(rng)
            else:
                raise ValueError(f"Unknown generation_mode: {generation_mode}")

            h = hash_sequence(inp)
            if h not in seen_hashes:
                seen_hashes.add(h)
                hashes_list.append(h)
                inputs_list.append(inp)
                step_targets_list.append(step_tgt)
                final_targets_list.append(final_tgt)

        if len(inputs_list) != size:
            raise RuntimeError(f"Requested size {size} not satisfied (generated {len(inputs_list)})")

        inputs_t = torch.from_numpy(np.stack(inputs_list, axis=0)).float()
        step_targets_t = torch.from_numpy(np.stack(step_targets_list, axis=0)).long()
        final_targets_t = torch.from_numpy(np.stack(final_targets_list, axis=0)).long()

        return SplitDataset(inputs_t, final_targets_t, step_targets_t, hashes_list), hashes_list

    def create_regime_b_splits(
        self,
        train_size: int = 6000,
        val_size: int = 1200,
        test_size: int = 1200,
        base_seed: int = 42,
    ) -> Tuple[SplitDataset, SplitDataset, SplitDataset]:
        """Creates the Stage 4 Regime B dataset splits with complete cryptographic verification."""
        train_set, train_hashes = self.generate_disjoint_split(
            train_size, seed=base_seed + 1000, generation_mode="regime_b_masked"
        )
        val_set, val_hashes = self.generate_disjoint_split(
            val_size, seed=base_seed + 50000, generation_mode="regime_b_masked"
        )
        test_set, test_hashes = self.generate_disjoint_split(
            test_size, seed=base_seed + 90000, generation_mode="regime_b_heldout_test"
        )

        # Cryptographic leak verification
        s_tr = set(train_hashes)
        s_va = set(val_hashes)
        s_te = set(test_hashes)

        if s_tr & s_va:
            raise RuntimeError(f"Data leakage detected! {len(s_tr & s_va)} sequences overlap between train and val.")
        if s_tr & s_te:
            raise RuntimeError(f"Data leakage detected! {len(s_tr & s_te)} sequences overlap between train and test.")
        if s_va & s_te:
            raise RuntimeError(f"Data leakage detected! {len(s_va & s_te)} sequences overlap between val and test.")

        return train_set, val_set, test_set

    def create_masked_campaign_splits(
        self,
        train_size: int = 6000,
        val_size: int = 1200,
        base_seed: int = 42,
        excluded_source_values: Tuple[int, ...] = (0, 3, 5),
    ) -> Tuple[SplitDataset, SplitDataset]:
        """Creates the registered TRAIN-MASKED / VAL-SEEN pools of the held-out campaign.

        ADD runtime sources are masked to ``excluded_source_values`` (registered
        default {0, 3, 5}); SET literals remain unrestricted. TRAIN/VAL
        sequence-disjointness is verified via SHA-256 hash manifests (metadata only).
        """
        train_set, train_hashes = self.generate_disjoint_split(
            train_size,
            seed=base_seed + 1000,
            generation_mode="regime_b_masked",
            excluded_source_values=excluded_source_values,
        )
        val_set, val_hashes = self.generate_disjoint_split(
            val_size,
            seed=base_seed + 50000,
            generation_mode="regime_b_masked",
            excluded_source_values=excluded_source_values,
        )
        assert_manifests_disjoint(
            build_hash_manifest(train_hashes, "TRAIN-MASKED"),
            build_hash_manifest(val_hashes, "VAL-SEEN"),
        )
        return train_set, val_set

    def create_proxy_dev_split(
        self,
        per_class: int = 120,
        base_seed: int = 42,
        max_attempts: int = 100,
    ) -> SplitDataset:
        """Creates the registered PROXY-DEV split with exact per-class balance.

        Every sequence contains exactly one designated ADD with runtime source 3 and
        the requested post-ADD target class; classes are generated with ``per_class``
        sequences each. Generation is bounded: each sequence call has ``max_attempts``
        attempts, and the uniqueness-retry loop is bounded at
        ``max_attempts * per_class`` attempts per class. Exhaustion raises a fatal
        ``RuntimeError`` (no silent sample reduction, no mask relaxation).
        """
        if per_class < 1:
            raise ValueError(f"per_class must be >= 1, got {per_class!r}")

        rng = np.random.default_rng(base_seed + 70000)
        seen_hashes: Set[str] = set()
        inputs_list: List[np.ndarray] = []
        step_targets_list: List[np.ndarray] = []
        final_targets_list: List[np.ndarray] = []
        hashes_list: List[str] = []

        for target_class in range(self.task.num_values):
            made = 0
            attempts = 0
            while made < per_class:
                attempts += 1
                if attempts > max_attempts * per_class:
                    raise RuntimeError(
                        f"Exceeded PROXY-DEV uniqueness attempts for target class {target_class} "
                        f"(requested {per_class}, generated {made})."
                    )
                inp, step_tgt, final_tgt = self.task.generate_proxy_dev_sequence(
                    rng, target_class, max_attempts=max_attempts
                )
                h = hash_sequence(inp)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                hashes_list.append(h)
                inputs_list.append(inp)
                step_targets_list.append(step_tgt)
                final_targets_list.append(final_tgt)
                made += 1

        if len(inputs_list) != per_class * self.task.num_values:
            raise RuntimeError(
                f"PROXY-DEV split size mismatch: expected {per_class * self.task.num_values}, "
                f"got {len(inputs_list)}"
            )

        inputs_t = torch.from_numpy(np.stack(inputs_list, axis=0)).float()
        step_targets_t = torch.from_numpy(np.stack(step_targets_list, axis=0)).long()
        final_targets_t = torch.from_numpy(np.stack(final_targets_list, axis=0)).long()

        return SplitDataset(inputs_t, final_targets_t, step_targets_t, hashes_list)
