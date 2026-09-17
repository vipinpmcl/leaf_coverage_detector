import numpy as np

def connected_components(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    neighbors = [
        (-1,-1), (-1,0), (-1,1),
        (0,-1),          (0,1),
        (1,-1),  (1,0),  (1,1)
    ]

    components = []

    for r in range(h):
        for c in range(w):
            if mask[r,c] == 0 or visited[r,c]:
                continue

            stack = [(r,c)]
            visited[r,c] = True
            pixels = []

            while stack:
                cr, cc = stack.pop()
                pixels.append((cr,cc))

                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h and 0 <= nc < w
                        and mask[nr,nc] == 1
                        and not visited[nr,nc]
                    ):
                        visited[nr,nc] = True
                        stack.append((nr,nc))

            rows = [p[0] for p in pixels]
            cols = [p[1] for p in pixels]

            components.append({
                "area_patches": len(pixels),
                "min_row": min(rows),
                "max_row": max(rows),
                "min_col": min(cols),
                "max_col": max(cols),
                "width_patches": max(cols) - min(cols) + 1,
                "height_patches": max(rows) - min(rows) + 1
            })

    return sorted(components, key=lambda x: x["area_patches"], reverse=True)
