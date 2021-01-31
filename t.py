class A():
    n = True
    def __init__(self):
        print(self.n)

class B(A):
    n = False

class C(A):
    n = 2


print(A())
print(B())
print(C())
