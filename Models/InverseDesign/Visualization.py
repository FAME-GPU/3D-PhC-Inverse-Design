import numpy as np
import matplotlib.pyplot as plt


# ======== Visualization Function ========
def visualize_sample(tensor, save_path=None, title="Generated Sample", is_processed=False, hide_background=False,
                     draw_bbox=False, point_size: int = 5):
    # Ensure correct data shape
    if tensor.dim() == 5:  # [1,1,60,60,60]
        data = tensor[0, 0].cpu().numpy()
    else:  # [1,60,60,60]
        data = tensor[0].cpu().numpy()

    # Add numerical range check
    print(
        f"Generated Data Stats: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}, std={data.std():.4f}")

    if is_processed:
        mask = data != 1  # Mask for non-background values
        if np.sum(mask) == 0:
            # If no non-background values, show all
            mask = np.ones_like(data, dtype=bool)
    else:
        # For raw data, clip outliers
        data = np.clip(data, 0, 20)
        if hide_background:
            # Hide background values (points with value 1)
            mask = data != 1
            if np.sum(mask) == 0:
                # If no non-background values, show all
                mask = np.ones_like(data, dtype=bool)
        else:
            mask = np.ones_like(data, dtype=bool)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = np.where(mask)
    values = data[mask]

    if len(values) == 0:
        print("Warning: No valid data points to visualize")
        # Create a simple 3D scatter plot for empty data
        ax.scatter([], [], [], c=[], cmap='viridis', s=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{title} (No valid data)")
    else:
        # Ensure valid min and max values
        vmin = np.min(values) if len(values) > 0 else 0
        vmax = np.max(values) if len(values) > 0 else 1

        # If all values are the same, set a different color range
        if vmin == vmax:
            vmin = vmin - 0.1
            vmax = vmax + 0.1

        scatter = ax.scatter(
            x, y, z,
            c=values,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            s=max(1, int(point_size))
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Value')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

    # Optional bounding box drawing
    if draw_bbox and len(values) > 0:
        xmin, xmax = int(x.min()), int(x.max())
        ymin, ymax = int(y.min()), int(y.max())
        zmin, zmax = int(z.min()), int(z.max())
        # Draw bounding box edges on 3D view (8 edges)
        corners = np.array([
            [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]
        ])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in edges:
            xs = [corners[a, 0], corners[b, 0]]
            ys = [corners[a, 1], corners[b, 1]]
            zs = [corners[a, 2], corners[b, 2]]
            ax.plot(xs, ys, zs, color='red', linewidth=1.0, alpha=0.9)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)