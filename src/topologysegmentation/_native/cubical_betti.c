/* Binary cubical Betti numbers for a 2D or 3D mask.

   Foreground cells that share a corner are one component (8-connected in 2D,
   26-connected in 3D). Voids are finite 6-connected pieces of the complement.
   The Euler characteristic uses the same closed cubes as Gudhi, so the counts
   match a superlevel diagram of a 0/1 image read at a threshold in (0, 1].
*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int64_t find_root(int32_t *parent, int64_t x) {
    int64_t root = x;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (parent[x] != (int32_t)root) {
        int32_t next = parent[x];
        parent[x] = (int32_t)root;
        x = next;
    }
    return root;
}

static void unite(int32_t *parent, int64_t a, int64_t b) {
    a = find_root(parent, a);
    b = find_root(parent, b);
    if (a != b) {
        parent[b] = (int32_t)a;
    }
}

static int link_if_set(int32_t *parent, int64_t i, int64_t j) {
    if (parent[j] >= 0) {
        unite(parent, i, j);
    }
    return 0;
}

static int betti2(const uint8_t *mask, int64_t height, int64_t width, int32_t *out) {
    int64_t count = height * width;
    int32_t *parent = (int32_t *)malloc((size_t)count * sizeof(int32_t));
    if (parent == NULL) {
        return -1;
    }
    for (int64_t i = 0; i < count; i++) {
        parent[i] = -1;
    }
    for (int64_t row = 0; row < height; row++) {
        for (int64_t col = 0; col < width; col++) {
            int64_t index = row * width + col;
            if (!mask[index]) {
                continue;
            }
            parent[index] = (int32_t)index;
            if (row > 0) {
                if (col > 0) {
                    link_if_set(parent, index, index - width - 1);
                }
                link_if_set(parent, index, index - width);
                if (col + 1 < width) {
                    link_if_set(parent, index, index - width + 1);
                }
            }
            if (col > 0) {
                link_if_set(parent, index, index - 1);
            }
        }
    }
    int64_t components = 0;
    int64_t faces = 0;
    for (int64_t i = 0; i < count; i++) {
        if (parent[i] >= 0) {
            faces += 1;
            if (find_root(parent, i) == i) {
                components += 1;
            }
        }
    }
    free(parent);

    int64_t edges = 0;
    for (int64_t row = 0; row <= height; row++) {
        for (int64_t col = 0; col < width; col++) {
            int above = row > 0 && mask[(row - 1) * width + col];
            int below = row < height && mask[row * width + col];
            edges += above || below;
        }
    }
    for (int64_t row = 0; row < height; row++) {
        for (int64_t col = 0; col <= width; col++) {
            int left = col > 0 && mask[row * width + (col - 1)];
            int right = col < width && mask[row * width + col];
            edges += left || right;
        }
    }
    int64_t vertices = 0;
    for (int64_t row = 0; row <= height; row++) {
        for (int64_t col = 0; col <= width; col++) {
            int on = 0;
            if (row > 0 && col > 0 && mask[(row - 1) * width + (col - 1)]) on = 1;
            if (row > 0 && col < width && mask[(row - 1) * width + col]) on = 1;
            if (row < height && col > 0 && mask[row * width + (col - 1)]) on = 1;
            if (row < height && col < width && mask[row * width + col]) on = 1;
            vertices += on;
        }
    }
    out[0] = (int32_t)components;
    out[1] = (int32_t)(components - (vertices - edges + faces));
    return 0;
}

static int64_t offset3(int64_t z, int64_t y, int64_t x, int64_t height, int64_t width) {
    return (z * height + y) * width + x;
}

static int betti3(const uint8_t *mask, int64_t depth, int64_t height, int64_t width, int32_t *out) {
    int64_t count = depth * height * width;
    int32_t *parent = (int32_t *)malloc((size_t)count * sizeof(int32_t));
    int32_t *background = (int32_t *)malloc((size_t)count * sizeof(int32_t));
    uint8_t *touches = (uint8_t *)malloc((size_t)count);
    if (parent == NULL || background == NULL || touches == NULL) {
        free(parent);
        free(background);
        free(touches);
        return -1;
    }
    for (int64_t i = 0; i < count; i++) {
        parent[i] = -1;
        background[i] = -1;
        touches[i] = 0;
    }
    static const int neighbor[13][3] = {
        {-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1}, {-1, 0, -1}, {-1, 0, 0},
        {-1, 0, 1},   {-1, 1, -1}, {-1, 1, 0},  {-1, 1, 1},  {0, -1, -1},
        {0, -1, 0},   {0, -1, 1},  {0, 0, -1},
    };
    for (int64_t z = 0; z < depth; z++) {
        for (int64_t y = 0; y < height; y++) {
            for (int64_t x = 0; x < width; x++) {
                int64_t index = offset3(z, y, x, height, width);
                if (!mask[index]) {
                    continue;
                }
                parent[index] = (int32_t)index;
                for (int k = 0; k < 13; k++) {
                    int64_t zz = z + neighbor[k][0];
                    int64_t yy = y + neighbor[k][1];
                    int64_t xx = x + neighbor[k][2];
                    if (zz < 0 || yy < 0 || xx < 0 || yy >= height || xx >= width) {
                        continue;
                    }
                    int64_t other = offset3(zz, yy, xx, height, width);
                    link_if_set(parent, index, other);
                }
            }
        }
    }
    int64_t components = 0;
    int64_t cubes = 0;
    for (int64_t i = 0; i < count; i++) {
        if (parent[i] >= 0) {
            cubes += 1;
            if (find_root(parent, i) == i) {
                components += 1;
            }
        }
    }
    free(parent);

    for (int64_t z = 0; z < depth; z++) {
        for (int64_t y = 0; y < height; y++) {
            for (int64_t x = 0; x < width; x++) {
                int64_t index = offset3(z, y, x, height, width);
                if (mask[index]) {
                    continue;
                }
                background[index] = (int32_t)index;
                if (z > 0) link_if_set(background, index, offset3(z - 1, y, x, height, width));
                if (y > 0) link_if_set(background, index, offset3(z, y - 1, x, height, width));
                if (x > 0) link_if_set(background, index, offset3(z, y, x - 1, height, width));
            }
        }
    }
    for (int64_t z = 0; z < depth; z++) {
        for (int64_t y = 0; y < height; y++) {
            for (int64_t x = 0; x < width; x++) {
                if (z != 0 && y != 0 && x != 0 && z + 1 != depth && y + 1 != height && x + 1 != width) {
                    continue;
                }
                int64_t index = offset3(z, y, x, height, width);
                if (background[index] >= 0) {
                    touches[find_root(background, index)] = 1;
                }
            }
        }
    }
    int64_t voids = 0;
    for (int64_t i = 0; i < count; i++) {
        if (background[i] >= 0 && find_root(background, i) == i && !touches[i]) {
            voids += 1;
        }
    }
    free(background);
    free(touches);

    int64_t faces = 0;
    int64_t edges = 0;
    int64_t vertices = 0;
    for (int64_t z = 0; z <= depth; z++) {
        for (int64_t y = 0; y < height; y++) {
            for (int64_t x = 0; x < width; x++) {
                int behind = z > 0 && mask[offset3(z - 1, y, x, height, width)];
                int ahead = z < depth && mask[offset3(z, y, x, height, width)];
                faces += behind || ahead;
            }
        }
    }
    for (int64_t z = 0; z < depth; z++) {
        for (int64_t y = 0; y <= height; y++) {
            for (int64_t x = 0; x < width; x++) {
                int behind = y > 0 && mask[offset3(z, y - 1, x, height, width)];
                int ahead = y < height && mask[offset3(z, y, x, height, width)];
                faces += behind || ahead;
            }
        }
    }
    for (int64_t z = 0; z < depth; z++) {
        for (int64_t y = 0; y < height; y++) {
            for (int64_t x = 0; x <= width; x++) {
                int behind = x > 0 && mask[offset3(z, y, x - 1, height, width)];
                int ahead = x < width && mask[offset3(z, y, x, height, width)];
                faces += behind || ahead;
            }
        }
    }
    for (int64_t z = 0; z <= depth; z++) {
        for (int64_t y = 0; y <= height; y++) {
            for (int64_t x = 0; x < width; x++) {
                int on = 0;
                for (int dz = 0; dz < 2; dz++) {
                    for (int dy = 0; dy < 2; dy++) {
                        int64_t zz = z - dz;
                        int64_t yy = y - dy;
                        if (zz < 0 || yy < 0 || zz >= depth || yy >= height) continue;
                        if (mask[offset3(zz, yy, x, height, width)]) on = 1;
                    }
                }
                edges += on;
            }
        }
    }
    for (int64_t z = 0; z <= depth; z++) {
        for (int64_t y = 0; y < height; y++) {
            for (int64_t x = 0; x <= width; x++) {
                int on = 0;
                for (int dz = 0; dz < 2; dz++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int64_t zz = z - dz;
                        int64_t xx = x - dx;
                        if (zz < 0 || xx < 0 || zz >= depth || xx >= width) continue;
                        if (mask[offset3(zz, y, xx, height, width)]) on = 1;
                    }
                }
                edges += on;
            }
        }
    }
    for (int64_t z = 0; z < depth; z++) {
        for (int64_t y = 0; y <= height; y++) {
            for (int64_t x = 0; x <= width; x++) {
                int on = 0;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int64_t yy = y - dy;
                        int64_t xx = x - dx;
                        if (yy < 0 || xx < 0 || yy >= height || xx >= width) continue;
                        if (mask[offset3(z, yy, xx, height, width)]) on = 1;
                    }
                }
                edges += on;
            }
        }
    }
    for (int64_t z = 0; z <= depth; z++) {
        for (int64_t y = 0; y <= height; y++) {
            for (int64_t x = 0; x <= width; x++) {
                int on = 0;
                for (int dz = 0; dz < 2; dz++) {
                    for (int dy = 0; dy < 2; dy++) {
                        for (int dx = 0; dx < 2; dx++) {
                            int64_t zz = z - dz;
                            int64_t yy = y - dy;
                            int64_t xx = x - dx;
                            if (zz < 0 || yy < 0 || xx < 0 || zz >= depth || yy >= height || xx >= width) {
                                continue;
                            }
                            if (mask[offset3(zz, yy, xx, height, width)]) on = 1;
                        }
                    }
                }
                vertices += on;
            }
        }
    }
    int64_t chi = vertices - edges + faces - cubes;
    out[0] = (int32_t)components;
    out[1] = (int32_t)(components + voids - chi);
    out[2] = (int32_t)voids;
    return 0;
}

int cubical_betti(const uint8_t *mask, int ndim, const int64_t *shape, int32_t *out) {
    if (mask == NULL || shape == NULL || out == NULL || (ndim != 2 && ndim != 3)) {
        return -2;
    }
    int64_t count = 1;
    for (int axis = 0; axis < ndim; axis++) {
        if (shape[axis] <= 0 || count > (INT64_C(0x7fffffff) / shape[axis])) {
            return -1;
        }
        count *= shape[axis];
    }
    if (ndim == 2) {
        return betti2(mask, shape[0], shape[1], out);
    }
    return betti3(mask, shape[0], shape[1], shape[2], out);
}
