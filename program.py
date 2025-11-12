
from instruction import Instruction
from typing import List, Tuple, Dict, Set
from collections import defaultdict

from memory_system import MemoryBank, MemoryType

class Program:
    def __init__(self, instructions: List[Instruction], max_program_length = 256) -> None:
        self.instructions = instructions
        self._effective_instructions = None
        self.max_program_length = max_program_length
    def execute(self, memory:MemoryBank):
        for instruction in self.instructions:
            instruction.execute(memory)

    def copy(self) -> 'Program':
        """Deep copy of program"""
        # Need to copy instructions too (they contain operation references)
        return Program([
            Instruction(
                operation=instr.operation,  # Operations are immutable, can share
                dest_type=instr.dest_type,
                dest_index=instr.dest_index,
                source_types=instr.source_types.copy(),
                source_indices=instr.source_indices.copy()
            )
            for instr in self.instructions
        ])
    
    def __len__(self) -> int:
        return len(self.instructions)
    
    def __repr__(self) -> str:
        return f"Program({len(self.instructions)} instructions)"

    def _build_write_map(self) -> Dict[Tuple[MemoryType, int], List[int]]:
        """
        Build a map from (register_type, register_index) -> [instruction_indices]
        
        Shows which instructions write to each register.
        Only the LAST write to a register matters for dependency.
        
        Returns:
            Dict mapping (MemoryType, index) to list of instruction indices that write to it
        """
        write_map = defaultdict(list)
        
        for idx, instruction in enumerate(self.instructions):
            dest_key = (instruction.dest_type, instruction.dest_index)
            write_map[dest_key].append(idx)
        
        return write_map
    
    def _get_last_writer(self, 
                         register: Tuple[MemoryType, int], 
                         before_idx: int,
                         write_map: Dict) -> int:
        """
        Find the last instruction that wrote to this register before the given index.
        
        Args:
            register: (MemoryType, index) tuple
            before_idx: Only consider instructions before this index
            write_map: Pre-computed write map
        
        Returns:
            Instruction index, or -1 if no writer found
        """
        writers = write_map.get(register, [])
        
        # Find the last writer before before_idx
        valid_writers = [w for w in writers if w < before_idx]
        
        if valid_writers:
            return max(valid_writers)  # Most recent writer
        return -1
    
    def find_effective_instructions(self, 
                                   output_registers: List[Tuple[MemoryType, int]]) -> Set[int]:
        """
        Find which instructions actually affect the output (intron removal).
        
        Uses backward dependency analysis:
        1. Start with output registers
        2. Find instructions that write to those registers
        3. For each effective instruction, find what it reads
        4. Recursively trace back, finding the last writer BEFORE the instruction that reads
        
        Args:
            output_registers: List of (MemoryType, index) tuples for outputs
                             Example: [(MemoryType.SCALAR, 9)] for action output
        
        Returns:
            Set of instruction indices that are effective (non-introns)
        """
        if not self.instructions:
            return set()
        
        # Build write map once
        write_map = self._build_write_map()
        
        # Track which instructions are effective
        effective = set()
        
        # Track which registers we need to trace and which instruction reads them
        # Format: (register, reader_idx) where reader_idx is the instruction that reads it
        # For output registers, reader_idx = len(instructions) (end of program)
        registers_to_trace = [
            (reg, len(self.instructions)) for reg in output_registers
        ]
        
        # Process until no more registers to trace
        while registers_to_trace:
            current_register, reader_idx = registers_to_trace.pop()
            
            # Find the last instruction that wrote to this register BEFORE the reader
            writer_idx = self._get_last_writer(
                current_register, 
                reader_idx,  # Use reader_idx instead of len(instructions)
                write_map
            )
            
            # If no writer found, it might be:
            # - An observation register (read-only)
            # - An evolvable constant (never written)
            # - A register that's never used
            if writer_idx == -1:
                continue
            
            # If we've already processed this instruction, skip
            if writer_idx in effective:
                continue
            
            # Mark this instruction as effective
            effective.add(writer_idx)
            
            # Add all registers this instruction reads to the worklist
            # The reader is the current writer_idx (the instruction we just marked effective)
            instruction = self.instructions[writer_idx]
            for src_type, src_idx in zip(instruction.source_types, 
                                         instruction.source_indices):
                # Only trace working registers (non-negative indices)
                # Observation registers (negative) are inputs, not written by program
                if src_idx >= 0:
                    registers_to_trace.append(((src_type, src_idx), writer_idx))
        
        return effective
    
    def get_introns(self, output_registers: List[Tuple[MemoryType, int]]) -> Set[int]:
        """
        Get instruction indices that are introns (don't affect output).
        
        Args:
            output_registers: List of output register specifications
        
        Returns:
            Set of instruction indices that are introns
        """
        effective = self.find_effective_instructions(output_registers)
        all_indices = set(range(len(self.instructions)))
        return all_indices - effective
    
    def remove_introns(self, output_registers: List[Tuple[MemoryType, int]]) -> 'Program':
        """
        Create a new program with introns removed.
        
        Args:
            output_registers: List of output register specifications
        
        Returns:
            New Program with only effective instructions
        """
        effective = self.find_effective_instructions(output_registers)
        effective_instructions = [
            self.instructions[i] 
            for i in sorted(effective)  # Maintain order
        ]
        return Program(effective_instructions)
    
    def get_effective_length(self, output_registers: List[Tuple[MemoryType, int]]) -> int:
        """
        Get the number of effective instructions.
        
        Useful for fitness metrics and bloat control.
        """
        return len(self.find_effective_instructions(output_registers))
    
    def get_intron_ratio(self, output_registers: List[Tuple[MemoryType, int]]) -> float:
        """
        Get the ratio of intron instructions to total instructions.
        
        Returns:
            Float in [0, 1] where 0 = no introns, 1 = all introns
        """
        if len(self.instructions) == 0:
            return 0.0
        introns = self.get_introns(output_registers)
        return len(introns) / len(self.instructions)
    
    # ==================== PRETTY PRINTING ====================
    
    def to_string(self, 
                  output_registers: List[Tuple[MemoryType, int]] = None,
                  show_introns: bool = True) -> str:
        """
        Pretty print the program, optionally marking introns.
        
        Args:
            output_registers: If provided, compute and show introns
            show_introns: If True, mark intron instructions
        
        Returns:
            Formatted string representation
        """
        if output_registers is None or not show_introns:
            # Simple listing
            lines = []
            for idx, instr in enumerate(self.instructions):
                lines.append(f"{idx:3d}: {instr}")
            return "\n".join(lines)
        
        # Show with intron marking
        effective = self.find_effective_instructions(output_registers)
        lines = []
        for idx, instr in enumerate(self.instructions):
            marker = " " if idx in effective else "X"
            lines.append(f"{marker} {idx:3d}: {instr}")
        
        lines.append(f"\nEffective: {len(effective)}/{len(self.instructions)} "
                    f"({len(effective)/len(self.instructions)*100:.1f}%)")
        
        return "\n".join(lines)

class MatrixProgram:
    """ MATRIX REPRESENTATIONN OF THIS"""
    pass 