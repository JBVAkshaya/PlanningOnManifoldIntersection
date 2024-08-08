from sympy.matrices import Matrix
from sympy import sin, cos
import sympy

q_0, q_1, q_2, q_3, x, y, z, yaw, ee = sympy.symbols('q_0 q_1 q_2 q_3 x y z yaw ee')

__1=cos(q_0)
__2=cos(q_1)
__3=(__1*__2)
__4=cos(q_2)
__5=sin(q_1)
__6=(__1*__5)
__7=sin(q_2)
__8=((__3*__4)-(__6*__7))
__9=cos(q_3)
__10=((__3*__7)+(__6*__4))
__11=sin(q_3)
__12=sin(q_0)
__13=(__12*__2)
__14=(__12*__5)
__15=((__13*__4)-(__14*__7))
__16=((__13*__7)+(__14*__4))
__17=((__5*__4)+(__2*__7))
__18=((__2*__4)-(__5*__7))
__19=0
__20=0.124
__21=0.024
__22=0.128

# Generate base to link5 transformation matrix
trans_mat = Matrix([[((__8*__9)-(__10*__11)), (-__12), ((__8*__11)+(__10*__9)), ((__20*__8)+(((__21*__3)+(__22*__6))+-0.08))], 
 [((__15*__9)-(__16*__11)), __1, ((__15*__11)+(__16*__9)), ((__20*__15)+((__21*__13)+(__22*__14)))], 
 [(-((__17*__9)+(__18*__11))), __19, ((__18*__9)-(__17*__11)), ((((__22*__2)-(__21*__5))+0.176)-(__20*__17))], 
 [__19, __19, __19, 1]])
print(trans_mat)
print("________________________________________")
print(sympy.simplify(trans_mat))

print("________________________________________")

T_base = Matrix([[cos(yaw), sin(yaw), 0., x],
                [sin(yaw), cos(yaw), 0., y],
                [0.,              0.,        1, z],
                [0.,              0.,        0., 1   ]])

print(sympy.simplify(T_base.multiply(trans_mat)))

print("________________________________________")

ee = Matrix([
            [ee],
            [0.0],
            [0.0],
            [1.0]])

### (Seems simplyfied mat)*ee is same as simplified (mat*ee).

print(sympy.simplify(T_base.multiply(trans_mat)).multiply(ee))
print("________________________________________")

print(sympy.simplify(T_base.multiply(trans_mat).multiply(ee)))
