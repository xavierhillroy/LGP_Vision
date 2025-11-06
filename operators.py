from dis import Instruction

import instruction_set


class GeneticOperators:
    def __init__(self, instruction_set: instruction_set) -> None:
        self.instruction_set = instruction_set
    def micro_mutate(instruction:Instruction, rate:float):
        # 4 parts of instruct op, dest, 
        pass