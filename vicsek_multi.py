# 입자의 type이 여러개인 vicsek model
# M: 입자의 type 수
# N_i: i-type 입자의 수
# Interaction matrix W_ij

# <theta_ni> = angle of sum_{m, j | distance closer than 1 } W_ij * v_mj
# theta_ni_new = <theta_ni> + eta
# v_ni = (cos(theta_ni), sin(theta_ni)), i-type의 n번째 입자 속도벡터
# 속력은 1, 상호작용 거리도 1

# %%
import taichi as ti
import math
import numpy as np
from matplotlib import pyplot as plt
import time

ti.init(arch=ti.gpu)

v = 0.03
R = 1.0

@ti.data_oriented
class VicsekModelMultiType:
    def __init__(self, Ns: list[int], L: float, W: np.ndarray):
        Ns_list = Ns
        self.M = len(Ns_list)
        self.N = int(np.sum(Ns_list))
        self.L = L
        self.W = ti.Matrix.field(self.M, self.M, dtype=ti.f32, shape=())

        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=self.N)
        self.angle = ti.field(dtype=ti.f32, shape=self.N)
        self.particle_type = ti.field(dtype=ti.i32, shape=self.N)
        self.avg_angle_sin = ti.field(dtype=ti.f32, shape=self.N)
        self.avg_angle_cos = ti.field(dtype=ti.f32, shape=self.N)

        self.Ns_field = ti.field(dtype=ti.i32, shape=self.M)
        self.Ns_py = Ns_list

        self.type_sum_cos = ti.field(dtype=ti.f32, shape=self.M)
        self.type_sum_sin = ti.field(dtype=ti.f32, shape=self.M)
        self.type_order_params = ti.field(dtype=ti.f32, shape=self.M)

        self.W.from_numpy(W)
        self.Ns_field.from_numpy(np.array(Ns_list, dtype=np.int32))

    @ti.kernel
    def init_particles(self):
        current_N = 0
        for type_idx in ti.static(range(self.M)):
            num_particles_of_type = self.Ns_field[type_idx]
            for i in range(current_N, current_N + num_particles_of_type):
                self.pos[i] = ti.Vector([ti.random() * self.L, ti.random() * self.L])
                self.angle[i] = (ti.random() - 0.5) * 2 * math.pi
                self.particle_type[i] = type_idx
            current_N += num_particles_of_type

    @ti.kernel
    def step(self, current_eta_arg: ti.f32):
        for i in self.avg_angle_sin:
            self.avg_angle_sin[i] = 0.0
            self.avg_angle_cos[i] = 0.0

        for i in range(self.N):
            type_i = self.particle_type[i]
            sum_w_sin = 0.0
            sum_w_cos = 0.0
            for j in range(self.N):
                dist_vec = self.pos[j] - self.pos[i]
                for k in ti.static(range(2)):
                    if dist_vec[k] > self.L / 2:
                        dist_vec[k] -= self.L
                    elif dist_vec[k] < -self.L / 2:
                        dist_vec[k] += self.L
                dist_sq = dist_vec.dot(dist_vec)

                if dist_sq <= R * R:
                    type_j = self.particle_type[j]
                    interaction_strength = self.W[None][type_i, type_j]
                    sum_w_sin += interaction_strength * ti.sin(self.angle[j])
                    sum_w_cos += interaction_strength * ti.cos(self.angle[j])

            if sum_w_sin != 0.0 or sum_w_cos != 0.0:
                 avg_angle = ti.atan2(sum_w_sin, sum_w_cos)
                 noise = (ti.random() - 0.5) * 2 * current_eta_arg
                 self.angle[i] = avg_angle + noise
            else:
                 noise = (ti.random() - 0.5) * 2 * current_eta_arg
                 self.angle[i] += noise

            vel = ti.Vector([ti.cos(self.angle[i]), ti.sin(self.angle[i])]) * v
            self.pos[i] += vel

            for k in ti.static(range(2)):
                if self.pos[i][k] < 0:
                    self.pos[i][k] += self.L
                elif self.pos[i][k] >= self.L:
                    self.pos[i][k] -= self.L

    @ti.kernel
    def calculate_order_parameter(self):
        for i in range(self.M):
            self.type_sum_cos[i] = 0.0
            self.type_sum_sin[i] = 0.0

        for i in range(self.N):
            p_type = self.particle_type[i]
            self.type_sum_cos[p_type] += ti.cos(self.angle[i])
            self.type_sum_sin[p_type] += ti.sin(self.angle[i])

        for i in range(self.M):
            if self.Ns_field[i] > 0:
                sum_cos = self.type_sum_cos[i]
                sum_sin = self.type_sum_sin[i]
                self.type_order_params[i] = ti.sqrt(sum_cos*sum_cos + sum_sin*sum_sin) / self.Ns_field[i]
            else:
                 self.type_order_params[i] = 0.0

    @ti.kernel
    def get_data_for_gui(self, positions_np: ti.types.ndarray(), types_np: ti.types.ndarray()):
        for i in range(self.N):
            for k in ti.static(range(2)):
                positions_np[i, k] = self.pos[i][k] / self.L
            types_np[i] = self.particle_type[i]

def simulate_multiple(
    Ns=[500, 500],
    W=np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32),
    L=10.0,
    eta=0.1,
    warm_up_steps=1000,
    steps_per_frame=5
):
    M = len(Ns)
    N = sum(Ns)
    model = VicsekModelMultiType(Ns, L, W)
    model.init_particles()

    print(f"Warm-up for {warm_up_steps} steps...")
    for _ in range(warm_up_steps):
        model.step(eta)
    print("Warm-up finished.")

    type_colors = [0x1f77b4, 0xff7f0e, 0x2ca02c, 0xd62728, 0x9467bd, 0x8c564b, 0xe377c2, 0x7f7f7f, 0xbcbd22, 0x17becf]
    if M > len(type_colors):
        np.random.seed(0)
        type_colors.extend([int(c) for c in np.random.randint(0, 0xFFFFFF, size=M - len(type_colors))])

    gui = ti.GUI("Multi-Type Vicsek Model", res=(512, 512), background_color=0xFFFFFF)
    positions_np = np.zeros((N, 2), dtype=np.float32)
    types_np = np.zeros(N, dtype=np.int32)
    colors_np = np.zeros(N, dtype=np.uint32)

    step_count = 0
    while gui.running:
        for _ in range(steps_per_frame):
            model.step(eta)
            step_count += 1

        model.calculate_order_parameter()
        model.get_data_for_gui(positions_np, types_np)

        for i in range(N):
            colors_np[i] = type_colors[types_np[i] % len(type_colors)]

        gui.circles(positions_np, radius=2, color=colors_np)

        gui.text(f"Step: {step_count}, Eta: {eta:.3f}", (0.02, 0.98), font_size=18, color=0x0)
        order_params = model.type_order_params.to_numpy()
        info_y = 0.94
        for i in range(M):
           gui.text(f"Type {i} (N={Ns[i]}) OP: {order_params[i]:.3f}", (0.02, info_y), font_size=16, color=type_colors[i % len(type_colors)])
           info_y -= 0.04

        gui.show()

if __name__ == "__main__":
    particle_counts_per_type = [1000, 1000]
    interaction_matrix = np.array([
        [1, -1],
        [-1, 1]
    ], dtype=np.float32)
    noise_level = 0.5

    simulate_multiple(
        Ns=particle_counts_per_type,
        W=interaction_matrix,
        eta=noise_level,
        steps_per_frame=4
    )