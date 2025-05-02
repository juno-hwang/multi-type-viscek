import taichi as ti, math, numpy as np
from matplotlib import pyplot as plt
import time

ti.init(arch=ti.gpu)

v = 0.03
R = 1.0

@ti.data_oriented
class VicsekModel:
    """Vicsek model with O(N) neighbor search using spatial hashing."""
    def __init__(self, N: int, L: float):
        self.N = N
        self.L = L
        self.v = v
        self.R = R

        self.cell_size = self.R
        self.grid_dim = int(math.ceil(self.L / self.cell_size))
        self.grid_size = self.grid_dim * self.grid_dim

        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.angle = ti.field(dtype=ti.f32, shape=N)

        self.avg_angle_sin = ti.field(dtype=ti.f32, shape=N)
        self.avg_angle_cos = ti.field(dtype=ti.f32, shape=N)
        self.neighbor_count = ti.field(dtype=ti.i32, shape=N)

        self.grid_head = ti.field(dtype=ti.i32, shape=self.grid_size)
        self.grid_next = ti.field(dtype=ti.i32, shape=N)

    @ti.kernel
    def init_particles(self):
        for i in range(self.N):
            self.pos[i] = ti.Vector([ti.random() * self.L, ti.random() * self.L])
            self.angle[i] = (ti.random() - 0.5) * 2 * math.pi

    @ti.kernel
    def _clear_accumulators_and_hash(self):
        for i in self.avg_angle_sin:
            self.avg_angle_sin[i] = 0.0
            self.avg_angle_cos[i] = 0.0
            self.neighbor_count[i] = 0
        for h in self.grid_head:
            self.grid_head[h] = -1

    @ti.kernel
    def _build_spatial_hash(self):
        cell_size = ti.static(self.cell_size)
        grid_dim = ti.static(self.grid_dim)

        for i in range(self.N):
            cx = int(self.pos[i][0] / cell_size) % grid_dim
            cy = int(self.pos[i][1] / cell_size) % grid_dim
            h = cx * grid_dim + cy

            self.grid_next[i] = self.grid_head[h]
            self.grid_head[h] = i

    @ti.kernel
    def _compute_and_move(self, current_eta: ti.f32):
        R = ti.static(self.R)
        R2 = R * R
        L = ti.static(self.L)
        cell_size = ti.static(self.cell_size)
        grid_dim = ti.static(self.grid_dim)

        for i in range(self.N):
            cx = int(self.pos[i][0] / cell_size)
            cy = int(self.pos[i][1] / cell_size)

            for dx in ti.static(range(-1, 2)):
                for dy in ti.static(range(-1, 2)):
                    ncx = (cx + dx + grid_dim) % grid_dim
                    ncy = (cy + dy + grid_dim) % grid_dim
                    h = ncx * grid_dim + ncy

                    j = self.grid_head[h]
                    while j != -1:
                        dist_vec = self.pos[j] - self.pos[i]
                        for k in ti.static(range(2)):
                            if dist_vec[k] > L / 2:
                                dist_vec[k] -= L
                            elif dist_vec[k] < -L / 2:
                                dist_vec[k] += L

                        if dist_vec.dot(dist_vec) <= R2:
                            self.avg_angle_sin[i] += ti.sin(self.angle[j])
                            self.avg_angle_cos[i] += ti.cos(self.angle[j])
                            self.neighbor_count[i] += 1
                        j = self.grid_next[j]

            if self.neighbor_count[i] > 0:
                avg_angle = ti.atan2(
                    self.avg_angle_sin[i] / self.neighbor_count[i],
                    self.avg_angle_cos[i] / self.neighbor_count[i]
                )
                noise = (ti.random() - 0.5) * 2 * current_eta
                self.angle[i] = avg_angle + noise
            else:
                self.angle[i] += (ti.random() - 0.5) * 2 * current_eta

            vel = ti.Vector([ti.cos(self.angle[i]), ti.sin(self.angle[i])]) * self.v
            self.pos[i] += vel

            for k in ti.static(range(2)):
                if self.pos[i][k] < 0:
                    self.pos[i][k] += L
                elif self.pos[i][k] > L:
                    self.pos[i][k] -= L

    def step(self, current_eta: float):
        self._clear_accumulators_and_hash()
        self._build_spatial_hash()
        self._compute_and_move(current_eta)

    @ti.kernel
    def calculate_order_parameter(self) -> ti.f32:
        sum_cos = 0.0
        sum_sin = 0.0
        for i in range(self.N):
            sum_cos += ti.cos(self.angle[i])
            sum_sin += ti.sin(self.angle[i])
        return ti.sqrt(sum_cos * sum_cos + sum_sin * sum_sin) / self.N

    @ti.kernel
    def get_positions(self, positions: ti.types.ndarray()):
        for i in range(self.N):
            positions[i, 0] = self.pos[i][0] / self.L
            positions[i, 1] = self.pos[i][1] / self.L

def run_vicsek_model(
    N=10000,
    L=1.0,
    eta_schedule=np.linspace(0.01, np.pi, 10000),
    verbose=False,
    warm_up_steps=1000
):
    max_steps = len(eta_schedule)
    model = VicsekModel(N, L)
    model.init_particles()

    initial_eta = eta_schedule[0]
    if verbose:
        print(f"Warming up for {warm_up_steps} steps with eta={initial_eta:.4f}")

    for _ in range(warm_up_steps):
        model.step(initial_eta)

    order_param_history = []
    current_step = 0

    while current_step < max_steps:
        current_eta = eta_schedule[current_step]
        model.step(current_eta)

        op = model.calculate_order_parameter()
        order_param_history.append(op)
        if verbose and current_step % 100 == 0:
            print(f"Step: {current_step}, Eta: {current_eta:.4f}, Order Param: {float(op):.4f}")

        current_step += 1

    return order_param_history

def run_phase_diagram(Ns: list[int], desired_time: float = 60.0):
    results = {}
    for N in Ns:
        rho = N / (10 * 10)
        start_time = time.time()
        plt.figure(figsize=(10, 5))
        list_of_history = []
        while time.time() - start_time < desired_time:
            order_param_history = run_vicsek_model(
                N=N,
                L=10,
                eta_schedule=np.linspace(0, np.pi, 10000),
                verbose=False,
                warm_up_steps=1000
            )
            end_time = time.time()
            print(f"Time taken for rho={rho}, N={N}, L={np.sqrt(N / rho):.2f}: {end_time - start_time:.2f} seconds")

            history_float = [float(op) for op in order_param_history]
            smoothing_window = 100
            smoothed_history = np.convolve(history_float, np.ones(smoothing_window) / smoothing_window, mode='valid')
            plt.plot(smoothed_history, alpha=0.2, color='black')
            list_of_history.append(smoothed_history)
        mean_history = np.mean(np.array(list_of_history), axis=0)
        plt.plot(mean_history, color='red')
        plt.xlabel('Time Steps')
        plt.ylabel('Order Parameter')
        plt.title(f'rho={rho}, N={N}, L={np.sqrt(N / rho):.2f}')
        plt.show()
        results[N] = mean_history
    return results

def simulate(N=1000, L=10.0, eta=0.1, warm_up_steps=1000, update_interval=1):
    gui = ti.GUI("Vicsek Model", res=(512, 512), background_color=0xFFFFFF)
    model = VicsekModel(N, L)
    model.init_particles()

    print(f"워밍업 단계: {warm_up_steps}회 반복")
    for _ in range(warm_up_steps):
        model.step(eta)

    positions = np.zeros((N, 2), dtype=np.float32)
    step_count = 0

    while gui.running:
        for _ in range(update_interval):
            model.step(eta)
            step_count += 1

        model.get_positions(positions)

        gui.text(f"N: {N}, eta: {eta:.4f}, Step: {step_count}", (0.05, 0.95), color=0x000000)
        gui.circles(positions, radius=2, color=0x000000)

        gui.show()

if __name__ == "__main__":
    order_param_history = simulate(
        N=10000, L=10.0, eta=0.5, update_interval=5
    )
    plt.plot(order_param_history)
    plt.title("Order parameter over time (spatial hashing)")
    plt.xlabel("Step")
    plt.ylabel("Order parameter")
    plt.show()