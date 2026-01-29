
import torch
import numpy as np

def build_theoretical_grid():
    values = [0.0]
    # Denormalized
    for m in range(1, 8):
        val = (2 ** -6) * (m / 8)
        values.append(val)
    # Normalized (e=1 to 14)
    for e in range(1, 15):
        for m in range(8):
            exp = e - 7
            mantissa = 1.0 + m / 8
            val = (2 ** exp) * mantissa
            values.append(val)
    # Extended (e=15 for E4M3FN)
    for m in range(8):
        val = (2 ** 8) * (1.0 + m / 8)
        if val <= 448:
            values.append(val)
    return sorted(set(values))

def build_native_grid():
    # Generate all possible byte values (0-255)
    all_bytes = torch.arange(256, dtype=torch.uint8)
    # Reinterpret as float8_e4m3fn
    fp8_vals = all_bytes.view(torch.float8_e4m3fn)
    # Cast back to float32 to see actual values
    f32_vals = fp8_vals.float()
    
    # Filter positive values and remove NaNs/Infs if any (E4M3FN has NaN but no Inf)
    # E4M3FN: 0x7F and 0xFF are NaN
    valid_vals = f32_vals[~f32_vals.isnan()]
    pos_vals = valid_vals[valid_vals >= 0]
    
    # Get unique values and sort
    return sorted(pos_vals.unique().tolist())

if __name__ == "__main__":
    if not hasattr(torch, 'float8_e4m3fn'):
        print("Error: torch.float8_e4m3fn not supported in this PyTorch version.")
        exit(1)

    theo = build_theoretical_grid()
    native = build_native_grid()
    
    print(f"Theoretical count: {len(theo)}")
    print(f"Native count:      {len(native)}")
    
    if len(theo) != len(native):
        print("Mismatch in count!")
    
    # Compare values with tolerance
    theo_t = torch.tensor(theo)
    native_t = torch.tensor(native)
    
    if len(theo_t) == len(native_t):
        diff = (theo_t - native_t).abs().max().item()
        print(f"Max difference: {diff:.10f}")
        if diff < 1e-6:
            print("Grids match perfectly within tolerance.")
        else:
            print("Grids differ!")
            # Find missing
            s_theo = set([round(x, 6) for x in theo])
            s_native = set([round(x, 6) for x in native])
            print(f"In output but not in native: {s_theo - s_native}")
            print(f"In native but not in output: {s_native - s_theo}")
    else:
        print("Cannot compare element-wise due to length mismatch.")
