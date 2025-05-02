# %%
import taichi as ti, math, numpy as np
from matplotlib import pyplot as plt
import time
ti.init(arch=ti.gpu)

v = 0.03
R = 1

@ti.data_oriented
class VicsekModel:
    def __init__(self, N, L):
        self.N = N
        self.L = L
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.angle = ti.field(dtype=ti.f32, shape=N)
        self.avg_angle_sin = ti.field(dtype=ti.f32, shape=N)
        self.avg_angle_cos = ti.field(dtype=ti.f32, shape=N)
        self.neighbor_count = ti.field(dtype=ti.i32, shape=N)

    @ti.kernel
    def init_particles(self):
        for i in range(self.N):
            self.pos[i] = ti.Vector([ti.random() * self.L, ti.random() * self.L])
            self.angle[i] = (ti.random() - 0.5) * 2 * math.pi

    @ti.kernel
    def step(self, current_eta_arg: ti.f32):
        for i in self.avg_angle_sin:
            self.avg_angle_sin[i] = 0.0
            self.avg_angle_cos[i] = 0.0
            self.neighbor_count[i] = 0
    
        for i in range(self.N):
            for j in range(self.N):
                dist_vec = self.pos[j] - self.pos[i]
                for k in ti.static(range(2)):
                    if dist_vec[k] > self.L / 2:
                        dist_vec[k] -= self.L
                    elif dist_vec[k] < -self.L / 2:
                        dist_vec[k] += self.L
                dist_sq = dist_vec.dot(dist_vec)
    
                if dist_sq <= R * R:
                    self.avg_angle_sin[i] += ti.sin(self.angle[j])
                    self.avg_angle_cos[i] += ti.cos(self.angle[j])
                    self.neighbor_count[i] += 1
    
        for i in range(self.N):
            if self.neighbor_count[i] > 0:
                avg_sin = self.avg_angle_sin[i] / self.neighbor_count[i]
                avg_cos = self.avg_angle_cos[i] / self.neighbor_count[i]
                avg_angle = ti.atan2(avg_sin, avg_cos)
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
                elif self.pos[i][k] > self.L:
                    self.pos[i][k] -= self.L

    @ti.kernel
    def calculate_order_parameter(self) -> ti.f32:
        sum_cos = 0.0
        sum_sin = 0.0
        for i in range(self.N):
            sum_cos += ti.cos(self.angle[i])
            sum_sin += ti.sin(self.angle[i])
    
        order_param = ti.sqrt(sum_cos * sum_cos + sum_sin * sum_sin) / self.N
        return order_param

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
        # 지정된 간격만큼 시뮬레이션 실행
        for _ in range(update_interval):
            model.step(eta)
            step_count += 1
        
        model.get_positions(positions)
        
        gui.text(f"N: {N}, eta: {eta:.4f}, Step: {step_count}", (0.05, 0.95), color=0x000000)
        gui.circles(positions, radius=2, color=0x000000)
        
        gui.show()

# %%

if __name__ == "__main__":
    Ns = [10 * 2**i for i in range(0, 11)]
    phase_diagram = run_phase_diagram(Ns)

    z = np.zeros((len(Ns), len(phase_diagram[Ns[0]])//500))
    for i, N in enumerate(Ns):
        for j in range(len(phase_diagram[N])//500):
            start_idx = j * 500
            end_idx = start_idx + 500
            z[i, j] = np.mean(phase_diagram[N][start_idx:end_idx])

    num_cols = len(phase_diagram[Ns[0]])//500  

    eta_values = np.linspace(0, np.pi, 10000)
    bin_centers = []
    for j in range(num_cols):
        start_idx = j * 500
        end_idx = start_idx + 500
        bin_centers.append(np.mean(eta_values[start_idx:end_idx]))

    plt.figure(figsize=(5, 4))
    im = plt.imshow(z, aspect='auto', interpolation='bilinear', cmap='viridis', origin='lower', 
                    extent=[0, np.pi, 0, len(Ns)-1])

    plt.xticks(np.linspace(0, np.pi, 5), [f"{v:.2f}" for v in np.linspace(0, np.pi, 5)])
    y_positions = np.arange(len(Ns))
    plt.yticks(y_positions, [f"{N}" for N in Ns])
    plt.xlabel('Eta (η)')
    plt.ylabel('N')
    plt.colorbar(im, label='Order Parameter')
    plt.tight_layout()
    plt.show()