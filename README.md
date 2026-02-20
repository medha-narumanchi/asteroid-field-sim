# Asteroid Field Gravity Simulation

Simulates 800 asteroids orbiting a planet under Newtonian gravity, using NVIDIA Warp for parallel computation and exported to USD for 3D viewing.

## What it does
- Runs physics on all 800 asteroids simultaneously using a Warp GPU kernel
- Colors and sizes asteroids based on their speed (red = fast, blue = slow)
- Exports the scene as a `.usda` file viewable in Blender or NVIDIA Omniverse
- Tracks energy conservation over time and plots it

## Physics
Standard Euler integration with Newtonian gravity:
```
a = GM / r²
v = v + a * dt
x = x + v * dt
```
Asteroids start with orbital velocities (`sqrt(GM/r) * 0.8`) for stable near-circular orbits.

## Simulation Stats
| Metric | Value |
|---|---|
| Asteroids | 800 |
| Steps | 200 |
| Avg speed | 1.590 units/s |
| Energy error | 0.1084 |

## Tech Stack
- **NVIDIA Warp** — parallel physics kernel
- **OpenUSD** — 3D scene export (Omniverse/Blender compatible)
- **Matplotlib** — visualization and energy plots
- **NumPy** — data handling

## How to Run
```bash
pip install warp-lang usd-core numpy matplotlib Pillow

python simulation.py    # runs physics, exports USD
python visualize.py     # generates images and animation
```

## Project Structure
```
├── simulation.py
├── visualize.py
├── asteroid_field.usda
├── asteroid_field.png
├── energy_conservation.png
└── asteroid_animation.gif
```
