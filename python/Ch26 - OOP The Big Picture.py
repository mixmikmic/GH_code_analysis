class C1:
    x = 1
    y = 2
    pass

class C2:
    x = 3
    z = 4

class C3(C1, C2):
    def print_c1_x(self):
        print(self.x)
        
    def print_c2_x(self):
        print(C2.x)

i3 = C3()
print("This is i3.x: ", i3.x)
i3.print_c1_x()
i3.print_c2_x()

