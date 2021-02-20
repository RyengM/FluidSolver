# euler lag collocated with heat driven, flip
import taichi as ti
import numpy as np
import random
ti.init(arch=ti.gpu)

res, dt = 512, 4e-3
t_decay = 0.95
nx = 256
dx = 1.0 / nx
block_capacity = 6
# temperature coefficient
alpha = 0.5
# gravity coefficient
beta = 0.35

# euler
mass = ti.field(float, shape=(nx, nx))
velocity = ti.Vector.field(2, float, shape=(nx, nx))
velocity_div = ti.field(float, shape=(nx, nx))
pressure = ti.field(float, shape=(nx, nx))
temperature = ti.field(float, shape=(nx, nx))
new_velocity = ti.Vector.field(2, float, shape=(nx, nx))
new_temperature = ti.field(float, shape=(nx, nx))
new_pressure = ti.field(float, shape=(nx, nx))

threshold = ti.field(float, shape=())

temperature_decay = 0.95
source_temperature = 500
source_density = 0.5

# particle
# GPU
particle_num = nx * nx * 6
p_mass = 0.5
p_density = ti.field(float, particle_num)
p_temperature = ti.field(float, particle_num)
p_position = ti.Vector.field(2, float, particle_num)
p_velocity = ti.Vector.field(2, float, particle_num)
p_C = ti.Matrix.field(2, 2, float, particle_num)

# CPU
# hash table, maintain particle index
particle_hash = []
particles = []
# init hash
for _ in range(nx):
    for _ in range(nx):
        particle_hash.append([])

# tracer
tracer_num = 10000
tracers = ti.Vector.field(2, float, tracer_num)
show_tracers = ti.Vector.field(2, float, tracer_num)
tracer_age = ti.field(float, tracer_num)

# 0: air, 1: fluid, 2: wall
grid_type = ti.field(int, shape=(nx, nx))
# 0: empty, 1: source
heat_type = ti.field(int, shape=(nx, nx))

class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur

pressure_pair = TexPair(pressure, new_pressure)

@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = max(0, min(nx - 1, I))
    return qf[I]

@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)

@ti.func
def bilerp(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)

@ti.func
def backtrace_rk1(vf: ti.template(), p, dt: ti.template()):
    p -= dt * bilerp(vf, p)
    return p


@ti.func
def backtrace_rk2(vf: ti.template(), p, dt: ti.template()):
    p_mid = p - 0.5 * dt * bilerp(vf, p)
    p -= dt * bilerp(vf, p_mid)
    return p


@ti.func
def backtrace_rk3(vf: ti.template(), p, dt: ti.template()):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt * v2
    v3 = bilerp(vf, p2)
    p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p

@ti.kernel
def copy(source: ti.template(), dest: ti.template()):
    for i, j in source:
        dest[i, j] = source[i, j]

# euler method
@ti.kernel
def advect_semilag(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace_rk3(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p)

@ti.kernel
def advect_tracer():
    for i in tracers:
        if tracer_age[i] > 0:
            p = tracers[i] * nx + 0.5
            p = backtrace_rk3(velocity, p, dt)
            tracers[i] += bilerp(velocity, p) * dt * dx

@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j).x
        vr = sample(vf, i + 1, j).x
        vb = sample(vf, i, j - 1).y
        vt = sample(vf, i, j + 1).y
        if i == 0:
            vl = 0
        if i == nx - 1:
            vr = 0
        if j == 0:
            vb = 0
        if j == nx - 1:
            vt = 0
        velocity_div[i, j] = (vr - vl + vt - vb) * 0.5

@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_div[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25

@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])

@ti.kernel
def p2g():
    # reset grid
    for i, j in mass:
        velocity[i, j] = [0 ,0]
        mass[i, j] = 0
        temperature[i, j] = 0
    # p2g
    for p in p_density:
        Xp = p_position[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        affine = p_mass * p_C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            velocity[base + offset] += weight * (p_mass * p_velocity[p] + affine @ dpos)
            mass[base + offset] += weight * p_mass
            temperature[base + offset] += weight * p_mass * p_temperature[p]
    for i, j in mass:
        if mass[i, j] > 0:
            velocity[i, j] /= mass[i, j]
            temperature[i, j] /= mass[i, j]

@ti.kernel
def g2p():
    for p in range(particle_num):
        Xp = p_position[p] * nx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        # new_rho = 0.0
        new_temperature = 0.0
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = velocity[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
            # new_rho += weight * mass[base + offset]
            new_temperature += weight * temperature[base + offset]
        p_velocity[p] = new_v
        p_C[p] = new_C
        # p_mass = new_rho
        p_temperature[p] = new_temperature

@ti.kernel
def add_heat():
    for i, j in velocity:
        # heat decay
        temperature[i, j] *= temperature_decay
        # add heat
        if heat_type[i, j] == 1:
            temperature[i, j] = source_temperature
            mass[i, j] = source_density

@ti.kernel
def add_force():
    for i, j in velocity:
        velocity[i, j].y += alpha * temperature[i, j] * dt - beta * mass[i, j] * dt
    # boundary
    for i, j in velocity:
        if grid_type[i, j] != 1:
            velocity[i, j] = [0, 0]

@ti.kernel
def set_grid_type():
    # 0: air, 1: fluid, 2: wall
    for i, j in velocity:
        grid_type[i, j] = 1
    for i in range(nx):
        grid_type[nx - 1, i] = 0
        grid_type[i, 0] = 0
        grid_type[i, nx - 1] = 0
        grid_type[0, i] = 2
    # set heat
    for i, j in velocity:
        if i >= nx // 2 - 2 and i <= nx // 2 + 2 and j >= 4 and j <= 6:
            heat_type[i, j] = 1

def init_particles():
    for i in range(nx):
        for j in range(nx):
            for _ in range(block_capacity):
                pos = [(i + random.random()) * dx, (j + random.random()) * dx]
                particles.append(pos)

def resample_particles():
    new_particles = []
    for i in range(nx):
        for j in range(nx):
            index = i * nx + j
            n = len(particle_hash[index])
            # copy(==) and delete(>)
            if n >= block_capacity:
                for k in range(block_capacity):
                    new_particles.append(particles[particle_hash[index][k]])
            # reseed
            elif (n < block_capacity):
                for k in range(n):
                    new_particles.append(particles[particle_hash[index][k]])
                for _ in range(block_capacity - n):
                    new_particles.append([(i + random.random()) * dx, (j + random.random()) * dx])
    particles.clear()
    for i in range(len(new_particles)):
        particles.append(new_particles[i])

def sort_particles():
    particle_hash.clear()
    # init hash
    for _ in range(nx):
        for _ in range(nx):
            particle_hash.append([])
    for i in range(len(particles)):
        pos = particles[i]
        px = int(pos[0] * nx)
        py = int(pos[1] * nx)
        index = px * nx + py
        if px >= nx - 1 or py >= nx - 1 or px < 1 or py < 1:
            continue
        particle_hash[index].append(i)

# should be replaced by higher precision
@ti.kernel
def move_particles():
    for p in range(particle_num):
        c1 = 2.0 / 9.0 * dt
        c2 = 3.0 / 9.0 * dt
        c3 = 4.0 / 9.0 * dt
        u = sample(velocity, p_position[p].x * nx, p_position[p].y * nx)
        mid = p_position[p] + 0.5 * u * dt
        u1 = sample(velocity, mid.x, mid.y)
        mid1 = p_position[p] + 0.75 * u1 * dt
        u2 = sample(velocity, mid1.x, mid1.y)
        # p_position[p] += (c1 * u + c2 * u1 + c3 * u2) * dx
        p_position[p] += u * dt * dx
        # p_position[p] = min(max(0, p_position[p], (nx - 1) * dx))

@ti.kernel
def init_tracers():
    for i in tracers:
        tracers[i] = [-1, -1]

@ti.kernel
def awake_tracer():
    threshold[None] += 0.000001
    for i in tracers:
        if tracer_age[i] == 0:
            if ti.random() < threshold[None]:
                tracer_age[i] += 0.001
                tracers[i] = [(nx // 2 - 2 + 4 * ti.random()) * dx, (4 + 2 * ti.random()) * dx]

def copy2GPU():
    temp = np.array(particles)
    p_position.from_numpy(temp)

# there may be wrong with the assignment
def copy2CPU():
    temp = p_position.to_numpy().tolist()
    for i in range(particle_num):
        particles[i] = temp[i]

def project():
    divergence(velocity)
    for _ in range(30):
        pressure_jacobi(pressure_pair.cur, pressure_pair.nxt)
        pressure_pair.swap()
    subtract_gradient(velocity, pressure_pair.cur)

def init():
    set_grid_type()
    init_particles()
    init_tracers()
    sort_particles()

def apic():
    resample_particles()
    copy2GPU()
    g2p()
    move_particles()
    p2g()
    copy2CPU()
    sort_particles()

def euler():
    advect_semilag(velocity, velocity, new_velocity)
    advect_semilag(velocity, temperature, new_temperature)
    copy(new_velocity, velocity)
    copy(new_temperature, temperature)

def step():
    awake_tracer()
    advect_tracer()
    euler()
    # apic()
    add_force()
    add_heat()
    project()

init()
paused = False
count = 0

gui = ti.GUI('euler', (res, res))
while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        e = gui.event
        if e.key == ti.GUI.ESCAPE:
            break
        elif e.key == 'p':
            paused = not paused

    if not paused:
        step()
        
    if count % 20 == 0:
        gui.circles(tracers.to_numpy(), radius=1.5, color=0x068587)
        filename = "results/output1/euler_lag_" + str(count) + ".png"
        gui.show(filename)
        print("output: " + str(count))
    count += 1