/* Vertex-complex grid and union-find persistence for Betti matching.

   The pairing follows the Python implementation: union by rank, births stored
   as the min (primal) or max (dual) of the root indices, and recorded births
   taken with the opposite reduction. Positive pairs compare filtration values.
   The loss still reads those values from the live tensor.
*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int32_t find_root(int32_t *parent, int32_t x) {
    int32_t root = x;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (parent[x] != root) {
        int32_t next = parent[x];
        parent[x] = root;
        x = next;
    }
    return root;
}

static void unite(int32_t *parent, int32_t *rank, int32_t *birth, int dual, int32_t x, int32_t y) {
    x = find_root(parent, x);
    y = find_root(parent, y);
    if (x == y) {
        return;
    }
    if (rank[x] > rank[y]) {
        parent[y] = x;
        birth[x] = dual ? (birth[x] > birth[y] ? birth[x] : birth[y])
                        : (birth[x] < birth[y] ? birth[x] : birth[y]);
    } else {
        parent[x] = y;
        birth[y] = dual ? (birth[x] > birth[y] ? birth[x] : birth[y])
                        : (birth[x] < birth[y] ? birth[x] : birth[y]);
        if (rank[x] == rank[y]) {
            rank[y] += 1;
        }
    }
}

static int positive_pair(double left, double right, int superlevel) {
    return superlevel ? left > right : left < right;
}

static void edge_boundary(const int32_t *index_map, int32_t cols, int32_t x, int32_t y, int32_t *left, int32_t *right) {
    if (x % 2 != 0) {
        *left = index_map[(x - 1) * cols + y];
        *right = index_map[(x + 1) * cols + y];
    } else {
        *left = index_map[x * cols + (y - 1)];
        *right = index_map[x * cols + (y + 1)];
    }
}

static void dual_boundary(
    const int32_t *index_map,
    int32_t cols,
    int32_t rows,
    int32_t sentinel,
    int32_t x,
    int32_t y,
    int32_t *left,
    int32_t *right
) {
    if (x % 2 == 0) {
        if (x == 0 || x == rows - 1) {
            *left = sentinel;
            /* A one-row grid has no square on either side of a horizontal edge. */
            if (x == 0 && x + 1 < rows) {
                *right = index_map[(x + 1) * cols + y];
            } else if (x == rows - 1 && x > 0) {
                *right = index_map[(x - 1) * cols + y];
            } else {
                *right = sentinel;
            }
        } else {
            *left = index_map[(x - 1) * cols + y];
            *right = index_map[(x + 1) * cols + y];
        }
    } else if (y == 0 || y == cols - 1) {
        *left = sentinel;
        if (y == 0 && y + 1 < cols) {
            *right = index_map[x * cols + 1];
        } else if (y == cols - 1 && y > 0) {
            *right = index_map[x * cols + (y - 1)];
        } else {
            *right = sentinel;
        }
    } else {
        *left = index_map[x * cols + (y - 1)];
        *right = index_map[x * cols + (y + 1)];
    }
}

static int stamp(
    int32_t *index_map,
    int32_t rows,
    int32_t cols,
    int32_t *coordinates,
    int32_t *edges,
    int32_t *counter_edges,
    int32_t *fine_rows,
    int32_t *fine_cols,
    int32_t *pixel_rows,
    int32_t *pixel_cols,
    int32_t *written,
    int32_t *counter,
    int32_t fine_row,
    int32_t fine_col,
    int32_t pixel_row,
    int32_t pixel_col,
    int is_edge
) {
    int32_t slot;
    if (fine_row < 0 || fine_col < 0 || fine_row >= rows || fine_col >= cols) {
        return 0;
    }
    if (index_map[fine_row * cols + fine_col] != -1) {
        return 0;
    }
    slot = *counter;
    index_map[fine_row * cols + fine_col] = slot;
    coordinates[slot * 2] = fine_row;
    coordinates[slot * 2 + 1] = fine_col;
    fine_rows[*written] = fine_row;
    fine_cols[*written] = fine_col;
    pixel_rows[*written] = pixel_row;
    pixel_cols[*written] = pixel_col;
    *written += 1;
    if (is_edge) {
        edges[*counter_edges] = slot;
        *counter_edges -= 1;
    }
    *counter -= 1;
    return 1;
}

int fill_vertex_grid(
    const int64_t *order,
    const double *keys,
    int32_t height,
    int32_t width,
    int32_t *index_map,
    int32_t rows,
    int32_t cols,
    int32_t *coordinates,
    int32_t *edges,
    int32_t num_edges,
    int32_t *fine_rows,
    int32_t *fine_cols,
    int32_t *pixel_rows,
    int32_t *pixel_cols,
    int32_t *n_writes
) {
    int32_t pixels = height * width;
    int32_t counter = rows * cols - 1;
    int32_t counter_edges = num_edges - 1;
    int32_t written = 0;
    int32_t start = 0;
    if (order == NULL || keys == NULL || index_map == NULL) {
        return -1;
    }
    while (start < pixels) {
        int32_t end = start + 1;
        double key = keys[start];
        int phase;
        while (end < pixels && keys[end] == key) {
            end += 1;
        }
        for (phase = 0; phase < 3; phase++) {
            int32_t cursor;
            for (cursor = start; cursor < end; cursor++) {
                int32_t flat = (int32_t)order[cursor];
                int32_t row = flat / width;
                int32_t col = flat - row * width;
                if (phase == 0) {
                    int k, l;
                    for (k = -1; k <= 1; k += 2) {
                        for (l = -1; l <= 1; l += 2) {
                            stamp(index_map, rows, cols, coordinates, edges, &counter_edges,
                                  fine_rows, fine_cols, pixel_rows, pixel_cols, &written, &counter,
                                  2 * row + k, 2 * col + l, row, col, 0);
                        }
                    }
                } else if (phase == 1) {
                    int offset;
                    for (offset = -1; offset <= 1; offset += 2) {
                        stamp(index_map, rows, cols, coordinates, edges, &counter_edges,
                              fine_rows, fine_cols, pixel_rows, pixel_cols, &written, &counter,
                              2 * row + offset, 2 * col, row, col, 1);
                        stamp(index_map, rows, cols, coordinates, edges, &counter_edges,
                              fine_rows, fine_cols, pixel_rows, pixel_cols, &written, &counter,
                              2 * row, 2 * col + offset, row, col, 1);
                    }
                } else {
                    stamp(index_map, rows, cols, coordinates, edges, &counter_edges,
                          fine_rows, fine_cols, pixel_rows, pixel_cols, &written, &counter,
                          2 * row, 2 * col, row, col, 0);
                }
            }
        }
        start = end;
    }
    *n_writes = written;
    return 0;
}

static int reverse_i32(int32_t *values, int32_t count) {
    int32_t left = 0;
    int32_t right = count - 1;
    while (left < right) {
        int32_t swap = values[left];
        values[left] = values[right];
        values[right] = swap;
        left += 1;
        right -= 1;
    }
    return 0;
}

static int compare_i32(const void *left, const void *right) {
    int32_t a = *(const int32_t *)left;
    int32_t b = *(const int32_t *)right;
    return (a > b) - (a < b);
}

int cubical_persistence(
    const int32_t *index_map,
    int32_t rows,
    int32_t cols,
    const int32_t *coordinates,
    const int32_t *edges,
    int32_t num_edges,
    const double *values,
    int32_t num_cubes,
    int superlevel,
    int mode,
    int32_t *h0,
    int32_t *n_h0,
    int32_t *h1,
    int32_t *n_h1,
    int32_t *columns,
    int32_t *n_columns,
    int32_t *critical,
    int32_t *n_critical,
    int32_t *columns2,
    int32_t *n_columns2
) {
    int32_t *parent = NULL;
    int32_t *rank = NULL;
    int32_t *birth = NULL;
    int32_t sentinel = num_cubes;
    int32_t dual_size = num_cubes + 1;
    int32_t count0 = 0;
    int32_t count1 = 0;
    int32_t count_columns = 0;
    int32_t count_critical = 0;
    int32_t count_columns2 = 0;
    int32_t edge_index;
    int32_t i;
    parent = (int32_t *)malloc((size_t)dual_size * sizeof(int32_t));
    rank = (int32_t *)malloc((size_t)dual_size * sizeof(int32_t));
    birth = (int32_t *)malloc((size_t)dual_size * sizeof(int32_t));
    if (parent == NULL || rank == NULL || birth == NULL) {
        free(parent);
        free(rank);
        free(birth);
        return -1;
    }
    for (i = 0; i < dual_size; i++) {
        parent[i] = i;
        rank[i] = 0;
        birth[i] = i;
    }
    for (edge_index = num_edges - 1; edge_index >= 0; edge_index--) {
        int32_t edge = edges[edge_index];
        int32_t x = coordinates[edge * 2];
        int32_t y = coordinates[edge * 2 + 1];
        int32_t left, right, root_left, root_right, born;
        dual_boundary(index_map, cols, rows, sentinel, x, y, &left, &right);
        root_left = find_root(parent, left);
        root_right = find_root(parent, right);
        if (root_left == root_right) {
            columns[count_columns++] = edge;
            continue;
        }
        born = birth[root_left] < birth[root_right] ? birth[root_left] : birth[root_right];
        if (mode == 1) {
            critical[count_critical++] = edge;
        }
        if (mode == 2) {
            columns2[count_columns2++] = born;
        }
        if (born < num_cubes && positive_pair(values[edge], values[born], superlevel)) {
            h1[count1 * 2] = edge;
            h1[count1 * 2 + 1] = born;
            count1 += 1;
        }
        unite(parent, rank, birth, 1, root_left, root_right);
    }
    reverse_i32(columns, count_columns);
    if (mode == 2) {
        qsort(columns2, (size_t)count_columns2, sizeof(int32_t), compare_i32);
    }

    for (i = 0; i < num_cubes; i++) {
        parent[i] = i;
        rank[i] = 0;
        birth[i] = i;
    }
    for (i = 0; i < count_columns; i++) {
        int32_t edge = columns[i];
        int32_t x = coordinates[edge * 2];
        int32_t y = coordinates[edge * 2 + 1];
        int32_t left, right, root_left, root_right, born;
        edge_boundary(index_map, cols, x, y, &left, &right);
        root_left = find_root(parent, left);
        root_right = find_root(parent, right);
        if (root_left == root_right) {
            continue;
        }
        born = birth[root_left] > birth[root_right] ? birth[root_left] : birth[root_right];
        if (born < num_cubes && positive_pair(values[born], values[edge], superlevel)) {
            h0[count0 * 2] = born;
            h0[count0 * 2 + 1] = edge;
            count0 += 1;
        }
        unite(parent, rank, birth, 0, root_left, root_right);
    }
    free(parent);
    free(rank);
    free(birth);
    *n_h0 = count0;
    *n_h1 = count1;
    *n_columns = count_columns;
    *n_critical = count_critical;
    *n_columns2 = count_columns2;
    return 0;
}

int image_persistence(
    const int32_t *index0,
    const int32_t *coord0,
    int32_t num0,
    const int32_t *index1,
    const int32_t *coord1,
    int32_t num1,
    int32_t rows,
    int32_t cols,
    const int32_t *columns1,
    int32_t n_columns1,
    const int32_t *critical0,
    int32_t n_critical0,
    int32_t *h0,
    int32_t *n_h0,
    int32_t *h1,
    int32_t *n_h1
) {
    int32_t dual_size = num0 > num1 + 1 ? num0 : num1 + 1;
    int32_t *parent = (int32_t *)malloc((size_t)dual_size * sizeof(int32_t));
    int32_t *rank = (int32_t *)malloc((size_t)dual_size * sizeof(int32_t));
    int32_t *birth = (int32_t *)malloc((size_t)dual_size * sizeof(int32_t));
    int32_t count0 = 0;
    int32_t count1 = 0;
    int32_t i;
    if (parent == NULL || rank == NULL || birth == NULL) {
        free(parent);
        free(rank);
        free(birth);
        return -1;
    }
    for (i = 0; i < num0; i++) {
        parent[i] = i;
        rank[i] = 0;
        birth[i] = i;
    }
    for (i = 0; i < n_columns1; i++) {
        int32_t edge = columns1[i];
        int32_t x = coord1[edge * 2];
        int32_t y = coord1[edge * 2 + 1];
        int32_t left, right, root_left, root_right, born;
        edge_boundary(index0, cols, x, y, &left, &right);
        root_left = find_root(parent, left);
        root_right = find_root(parent, right);
        if (root_left == root_right) {
            continue;
        }
        born = birth[root_left] > birth[root_right] ? birth[root_left] : birth[root_right];
        h0[count0 * 2] = born;
        h0[count0 * 2 + 1] = edge;
        count0 += 1;
        unite(parent, rank, birth, 0, root_left, root_right);
    }
    for (i = 0; i < dual_size; i++) {
        parent[i] = i;
        rank[i] = 0;
        birth[i] = i;
    }
    for (i = 0; i < n_critical0; i++) {
        int32_t edge = critical0[i];
        int32_t x = coord0[edge * 2];
        int32_t y = coord0[edge * 2 + 1];
        int32_t left, right, root_left, root_right, born;
        dual_boundary(index1, cols, rows, num1, x, y, &left, &right);
        root_left = find_root(parent, left);
        root_right = find_root(parent, right);
        if (root_left == root_right) {
            continue;
        }
        born = birth[root_left] < birth[root_right] ? birth[root_left] : birth[root_right];
        h1[count1 * 2] = edge;
        h1[count1 * 2 + 1] = born;
        count1 += 1;
        unite(parent, rank, birth, 1, root_left, root_right);
    }
    free(parent);
    free(rank);
    free(birth);
    *n_h0 = count0;
    *n_h1 = count1;
    return 0;
}
