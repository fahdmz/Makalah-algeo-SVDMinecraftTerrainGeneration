"""
SVD-Based Terrain Analysis for Procedurally Generated Heightmaps
Author: [Your Name]
NIM: [Your NIM]

This script demonstrates Singular Value Decomposition (SVD) applied to
Minecraft-style terrain heightmaps for smoothing and compression analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from noise import pnoise2
import time


TERRAIN_SIZE = 256  # 256x256 blocks
HEIGHT_MIN = 64     # Minimum terrain height (blocks)
HEIGHT_MAX = 200    # Maximum terrain height (blocks)
SEED = 42          # Random seed for reproducibility

# Perlin noise parameters (similar to Minecraft)
NOISE_SCALE = 50.0
NOISE_OCTAVES = 6
NOISE_PERSISTENCE = 0.5
NOISE_LACUNARITY = 2.0

# SVD test parameters
K_VALUES = [5, 10, 20, 30, 50, 100]  # Different compression levels to test

# ============================================================================
# TERRAIN GENERATION
# ============================================================================

def generate_perlin_terrain(size, scale, octaves, persistence, lacunarity, seed=0):
    """
    Generate terrain using Perlin noise (similar to Minecraft's algorithm)
    
    Parameters:
        size: Terrain size (size x size)
        scale: Controls feature size (larger = broader features)
        octaves: Number of noise layers (more = more detail)
        persistence: How much each octave contributes (0-1)
        lacunarity: How much detail increases per octave (typically 2.0)
        seed: Random seed for reproducibility
    
    Returns:
        2D numpy array of heights
    """
    terrain = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            # Generate Perlin noise value
            noise_val = pnoise2(
                i / scale,
                j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=seed
            )
            terrain[i][j] = noise_val
    
    # Normalize to Minecraft height range
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    terrain = terrain * (HEIGHT_MAX - HEIGHT_MIN) + HEIGHT_MIN
    
    return terrain

# ============================================================================
# SVD OPERATIONS
# ============================================================================

def apply_svd(heightmap):
    """
    Apply full SVD decomposition to heightmap
    
    Returns:
        U, S, Vt: SVD components
    """
    U, S, Vt = np.linalg.svd(heightmap, full_matrices=False)
    return U, S, Vt

def reconstruct_from_svd(U, S, Vt, k):
    """
    Reconstruct heightmap using only top k singular values
    
    Parameters:
        U, S, Vt: SVD components
        k: Number of components to keep
    
    Returns:
        Reconstructed heightmap
    """
    S_truncated = np.zeros_like(S)
    S_truncated[:k] = S[:k]
    reconstructed = U @ np.diag(S_truncated) @ Vt
    return reconstructed

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_metrics(original, reconstructed):
    """
    Calculate quality metrics for terrain reconstruction
    """
    # Mean Squared Error
    mse = np.mean((original - reconstructed) ** 2)
    
    # Peak Signal-to-Noise Ratio
    max_val = np.max(original)
    psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # Smoothness (average gradient magnitude)
    grad_y_orig, grad_x_orig = np.gradient(original)
    grad_y_recon, grad_x_recon = np.gradient(reconstructed)
    
    smoothness_orig = np.mean(np.sqrt(grad_x_orig**2 + grad_y_orig**2))
    smoothness_recon = np.mean(np.sqrt(grad_x_recon**2 + grad_y_recon**2))
    smoothness_ratio = smoothness_recon / smoothness_orig
    
    # Relative error
    relative_error = np.linalg.norm(original - reconstructed) / np.linalg.norm(original) * 100
    
    return {
        'mse': mse,
        'psnr': psnr,
        'smoothness_ratio': smoothness_ratio,
        'relative_error': relative_error
    }

def calculate_compression_ratio(size, k):
    """
    Calculate compression ratio for SVD with k components
    
    Original: size x size matrix (floats)
    Compressed: U[:, :k] + S[:k] + Vt[:k, :] 
              = (size*k + k + k*size) floats
    """
    original_size = size * size
    compressed_size = size * k + k + k * size
    return original_size / compressed_size

def calculate_cumulative_energy(S):
    """
    Calculate cumulative energy preserved by k components
    """
    total_energy = np.sum(S**2)
    cumulative = np.cumsum(S**2) / total_energy * 100
    return cumulative

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_terrain_3d(terrain, title, ax, colormap='terrain'):
    """
    Plot 3D terrain surface
    """
    size = terrain.shape[0]
    X, Y = np.meshgrid(range(size), range(size))
    
    surf = ax.plot_surface(X, Y, terrain, cmap=colormap, 
                           linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel('X (blocks)')
    ax.set_ylabel('Z (blocks)')
    ax.set_zlabel('Height (blocks)')
    ax.set_title(title)
    ax.set_zlim(HEIGHT_MIN - 10, HEIGHT_MAX + 10)
    
    # Set viewing angle
    ax.view_init(elev=30, azim=45)

def plot_singular_values(S, ax):
    """
    Plot singular values in log scale
    """
    ax.plot(range(1, len(S) + 1), S, 'b-', linewidth=2)
    ax.set_xlabel('Component Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Value Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=30, color='r', linestyle='--', alpha=0.5, label='k=30')
    ax.legend()

def plot_cumulative_energy(cumulative, ax):
    """
    Plot cumulative energy preservation
    """
    ax.plot(range(1, len(cumulative) + 1), cumulative, 'g-', linewidth=2)
    ax.axhline(y=95, color='r', linestyle='--', alpha=0.5, label='95% threshold')
    ax.set_xlabel('Number of Components (k)')
    ax.set_ylabel('Cumulative Energy (%)')
    ax.set_title('Energy Preservation vs Components')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Find k for 95% energy
    k_95 = np.argmax(cumulative >= 95) + 1
    ax.plot(k_95, 95, 'ro', markersize=8)
    ax.text(k_95 + 5, 93, f'k={k_95}', fontsize=10)

def plot_comparison_grid(original, reconstructions, k_values):
    """
    Plot grid of terrain comparisons
    """
    n = len(k_values) + 1
    cols = 3
    rows = (n + cols - 1) // cols
    
    fig = plt.figure(figsize=(15, 5 * rows))
    
    # Original
    ax = fig.add_subplot(rows, cols, 1, projection='3d')
    plot_terrain_3d(original, 'Original Terrain', ax)
    
    # Reconstructions
    for idx, (k, recon) in enumerate(zip(k_values, reconstructions)):
        ax = fig.add_subplot(rows, cols, idx + 2, projection='3d')
        plot_terrain_3d(recon, f'SVD (k={k})', ax)
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 70)
    print("SVD-Based Terrain Analysis")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: Generate Terrain
    # ========================================================================
    print(f"\n[1/5] Generating {TERRAIN_SIZE}x{TERRAIN_SIZE} terrain...")
    start_time = time.time()
    
    terrain = generate_perlin_terrain(
        TERRAIN_SIZE, 
        NOISE_SCALE, 
        NOISE_OCTAVES,
        NOISE_PERSISTENCE, 
        NOISE_LACUNARITY,
        SEED
    )
    
    gen_time = time.time() - start_time
    print(f"   ✓ Terrain generated in {gen_time:.2f}s")
    print(f"   Height range: {terrain.min():.1f} to {terrain.max():.1f} blocks")
    print(f"   Mean height: {terrain.mean():.1f} blocks")
    print(f"   Std deviation: {terrain.std():.1f} blocks")
    
    # ========================================================================
    # STEP 2: Apply SVD
    # ========================================================================
    print(f"\n[2/5] Applying SVD decomposition...")
    start_time = time.time()
    
    U, S, Vt = apply_svd(terrain)
    
    svd_time = time.time() - start_time
    print(f"   ✓ SVD computed in {svd_time:.2f}s")
    print(f"   Matrix rank: {len(S)}")
    print(f"   Top 5 singular values: {S[:5]}")
    
    # ========================================================================
    # STEP 3: Reconstruct with Different k Values
    # ========================================================================
    print(f"\n[3/5] Reconstructing terrain with different k values...")
    
    reconstructions = []
    metrics_table = []
    
    for k in K_VALUES:
        start_time = time.time()
        recon = reconstruct_from_svd(U, S, Vt, k)
        recon_time = time.time() - start_time
        
        metrics = calculate_metrics(terrain, recon)
        comp_ratio = calculate_compression_ratio(TERRAIN_SIZE, k)
        
        reconstructions.append(recon)
        metrics_table.append({
            'k': k,
            'time': recon_time,
            'compression': comp_ratio,
            **metrics
        })
        
        print(f"   k={k:3d}: Error={metrics['relative_error']:5.2f}%, "
              f"PSNR={metrics['psnr']:5.1f}dB, "
              f"Compression={comp_ratio:.1f}x")
    
    # ========================================================================
    # STEP 4: Analyze Results
    # ========================================================================
    print(f"\n[4/5] Analyzing results...")
    
    cumulative_energy = calculate_cumulative_energy(S)
    k_95 = np.argmax(cumulative_energy >= 95) + 1
    k_99 = np.argmax(cumulative_energy >= 99) + 1
    
    print(f"   Energy preservation:")
    print(f"      95% energy requires k={k_95} components")
    print(f"      99% energy requires k={k_99} components")
    print(f"   Compression potential:")
    print(f"      k={k_95} provides {calculate_compression_ratio(TERRAIN_SIZE, k_95):.1f}x compression")
    
    # ========================================================================
    # STEP 5: Generate Visualizations
    # ========================================================================
    print(f"\n[5/5] Generating visualizations...")
    
    # Figure 1: Singular value analysis
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_singular_values(S[:100], axes[0])
    plot_cumulative_energy(cumulative_energy[:100], axes[1])
    plt.tight_layout()
    plt.savefig('svd_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: svd_analysis.png")
    
    # Figure 2: Terrain comparisons
    fig2 = plot_comparison_grid(terrain, reconstructions, K_VALUES)
    plt.savefig('terrain_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: terrain_comparison.png")
    
    # Figure 3: Metrics visualization
    fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Error vs k
    axes[0, 0].bar([m['k'] for m in metrics_table], 
                   [m['relative_error'] for m in metrics_table],
                   color='steelblue')
    axes[0, 0].set_xlabel('k (components)')
    axes[0, 0].set_ylabel('Relative Error (%)')
    axes[0, 0].set_title('Reconstruction Error vs k')
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR vs k
    axes[0, 1].plot([m['k'] for m in metrics_table], 
                    [m['psnr'] for m in metrics_table],
                    'o-', color='green', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('k (components)')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('Quality (PSNR) vs k')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Compression vs k
    axes[1, 0].plot([m['k'] for m in metrics_table], 
                    [m['compression'] for m in metrics_table],
                    's-', color='orange', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('k (components)')
    axes[1, 0].set_ylabel('Compression Ratio')
    axes[1, 0].set_title('Compression Ratio vs k')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Smoothness ratio vs k
    axes[1, 1].plot([m['k'] for m in metrics_table], 
                    [m['smoothness_ratio'] for m in metrics_table],
                    'd-', color='purple', linewidth=2, markersize=8)
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('k (components)')
    axes[1, 1].set_ylabel('Smoothness Ratio')
    axes[1, 1].set_title('Smoothness vs k (lower = smoother)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: metrics_analysis.png")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Terrain size: {TERRAIN_SIZE}x{TERRAIN_SIZE} blocks")
    print(f"Total components: {len(S)}")
    print(f"\nRecommended settings:")
    print(f"  • For high quality: k={k_99} (99% energy, "
          f"{calculate_compression_ratio(TERRAIN_SIZE, k_99):.1f}x compression)")
    print(f"  • For balanced: k={k_95} (95% energy, "
          f"{calculate_compression_ratio(TERRAIN_SIZE, k_95):.1f}x compression)")
    print(f"  • For max smoothing: k=10 (minimal features, "
          f"{calculate_compression_ratio(TERRAIN_SIZE, 10):.1f}x compression)")
    
    print("\nGenerated files:")
    print("  • svd_analysis.png - Singular values and energy")
    print("  • terrain_comparison.png - Visual comparisons")
    print("  • metrics_analysis.png - Quantitative metrics")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    plt.show()

# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    main()