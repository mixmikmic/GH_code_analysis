from collections import defaultdict, deque


class Jump(Exception):
    def __init__(self, offset):
        self.offset = offset
        super().__init__(offset)

        
def opcode(operands):
    def decorator(f):
        class Opcode:
            def __set_name__(self, owner, name):
                self.opcode = name[3:]
                owner.opcodes[self.opcode] = self

            def __repr__(self):
                return f'<opcode {self.opcode} {operands!r}>'
            
            def value(self, operand, type_):
                if type_ == 'r':
                    return operand
                try:
                    return int(operand)
                except ValueError:
                    return self.registers[operand]

            def __call__(self, cpu, *ops):
                self.registers = cpu.registers
                try:
                    result = f(cpu, *map(self.value, ops, operands))
                    cpu.pos += 1
                except Jump as j:
                    cpu.pos += j.offset
                    result = None
                return self.opcode

        return Opcode()
    return decorator


class Proc:
    opcodes = {}
    
    def __init__(self, debug=True):
        self.reset(debug)
        
    def reset(self, debug=True):
        self.registers = dict.fromkeys('abcdefgh', 0)
        self.debug = debug
        if not debug:
            self.registers['a'] = 1
        self.pos = 0
    
    def run(self, instructions):
        if not self.debug:
            instructions = self.optimise(instructions)
        while 0 <= self.pos < len(instructions):
            opcode, *ops = instructions[self.pos]
            yield self.opcodes[opcode](self, *ops)

    @opcode('')
    def op_nop(self):
        pass
    
    @opcode('rv')
    def op_set(self, x, y):
        self.registers[x] = y
    
    @opcode('rv')
    def op_sub(self, x, y):
        self.registers[x] -= y

    @opcode('rv')
    def op_mul(self, x, y):
        self.registers[x] *= y
        
    @opcode('rv')
    def op_mod(self, x, y):
        self.registers[x] %= y
    
    @opcode('vv')
    def op_jnz(self, x, y):
        if x:
            raise Jump(y)
    
    def optimise(self, instructions):
        # modulus operation over two registers, setting a third flag register
        # using two working registers. If the flag register
        # is set, jump out of the outer loop
        operand1, operand2 = instructions[13][2], instructions[11][2]
        workreg = instructions[11][1]
        flagreg = instructions[15][1]
        return instructions[:10] + [
            ('set', workreg, operand1),
            ('mod', workreg, operand2),
            ('jnz', workreg, '8'),
            ('set', flagreg, '0'),
            ('jnz', '1', '11'),
            ('jnz', '1', '5'),
        ] + [('nop',)] * 4 + instructions[20:]

with open('inputs/day23.txt') as day23:
    instructions = [line.split() for line in day23]

proc = Proc()
print('Part 1:', sum(1 for opcode in proc.run(instructions) if opcode == 'mul'))

proc = Proc(debug=False)
deque(proc.run(instructions), 0)
print('Part 2:', proc.registers['h'])

# Cheating option, just run Python code
lower = (99 * 100) + 100000
upper = lower + 17000
h = 0
for b in range(lower, upper + 1, 17):
    f = 1
    for d in range(2, b):
        if b % d == 0:
            f = 0
            break
    if not f:
        h += 1
print(h)

