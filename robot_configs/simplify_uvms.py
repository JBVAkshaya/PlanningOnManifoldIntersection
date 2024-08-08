from sympy.matrices import Matrix
from sympy import sin, cos
import sympy

q1, q2, x, y, z, yaw, l1, l2, bx, bz = sympy.symbols('q_1 q_2 x y z yaw l1 l2 bx bz')

T_f_to_rovbase = Matrix([[cos(yaw), sin(yaw), 0., x],
                            [sin(yaw), cos(yaw), 0., y],
                            [0.,              0.,        1, z],
                            [0.,              0.,        0., 1   ]])

T_rovbase_to_armbase = Matrix([[1, 0., 0., bx/2.0],
                            [0., 1, 0., 0. ],
                            [0., 0., 1, bz/2.0],
                            [0., 0., 0., 1 ]])

T_armbase_to_arm_0 = Matrix(
                            [[1,  0.,  0., 0. ],
                            [0.,  0.,  1, 0. ],
                            [0.,  -1, 0., 0. ],
                            [0.,  0.,  0., 1 ]])    

T_arm_0_to_arm_1 = Matrix(
                            [[cos(q1), -sin(q1), 0., 0.],
                            [sin(q1), cos(q1),  0., 0.],
                            [0.,          0.,           1 , 0.],
                            [0.,          0.,           0., 1 ]])

T_arm_1_to_arm_2 = Matrix([
                            [cos(q2), -sin(q2), 0., l1],
                            [sin(q2), cos(q2),  0., 0.],
                            [0.,          0.,           1 , 0.],
                            [0.,          0.,           0., 1 ]])
end_effector = Matrix([
                        [l2],
                        [0.0],
                        [0.0],
                        [1.0]])


transform = T_f_to_rovbase.multiply(T_rovbase_to_armbase).multiply(T_armbase_to_arm_0).multiply(T_arm_0_to_arm_1).multiply(T_arm_1_to_arm_2)
position = transform.multiply(end_effector)

    
# Generate base to link5 transformation matrix

print(transform)
print("________________________________________")
print(sympy.simplify(transform))
print("________________________________________")
print(sympy.simplify(position))

