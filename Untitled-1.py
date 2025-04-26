class A:
    def show(self):
        print("A")

class B(A):
    def show(self):
        print("B 开始")
        super().show()  # 调用 A.show
        print("B 结束")

class C(A):
    def show(self):
        print("C 开始")
        super().show()  # 调用 A.show
        print("C 结束")

class D(B, C):
    def show(self):
        super().show()  # 按 MRO 顺序调用 B → C → A

d = D()
d.show()