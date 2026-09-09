from models import rimless_wheel as model

params = model.generate_params()
fixed_point = model.find_fixed_point(params)

perturbation = 1e-4
velocity_minus = fixed_point - perturbation
velocity_plus = fixed_point + perturbation

mapped_minus = model.poincare_map(velocity_minus, params)
mapped_plus = model.poincare_map(velocity_plus, params)

floquet_multiplier = (mapped_plus - mapped_minus) / (velocity_plus - velocity_minus)

print(f"Fixed point: v* = {fixed_point:.6f} rad/s")
print(f"f(v* - eps) = {mapped_minus:.6f}, f(v* + eps) = {mapped_plus:.6f}")
print(f"Floquet multiplier (local slope of return map): {floquet_multiplier:.6f}")
print(f"|multiplier| < 1 -> locally stable: {abs(floquet_multiplier) < 1}")
