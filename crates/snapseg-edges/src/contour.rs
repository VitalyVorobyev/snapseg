//! Marching-squares contour extraction from a boolean mask.
//!
//! Input is a row-major `Array2<bool>` of shape `[H, W]` where `true` means
//! foreground. Output is one or more ordered closed polylines tracing the
//! foreground/background boundary at cell-edge midpoints (half-pixel
//! positions).
//!
//! Algorithm: classify every 2x2 cell of pixel centres into one of 16 cases,
//! emit 0/1/2 segments per cell whose endpoints lie at edge midpoints, then
//! walk the resulting graph to recover closed polylines.
//!
//! Saddle cases (0b0101 and 0b1010) are resolved by averaging the four
//! corner values: `>= 0.5` connects inside through the cell, otherwise
//! outside connects. For a pure boolean mask the average is exactly 0.5,
//! and the `>=` branch is chosen deterministically (treat saddles as
//! "inside connects"). This biases consistently and yields a closed
//! topology.
//!
//! Output orientation is **not guaranteed** to be CCW; refinement is
//! invariant to traversal direction (see `refine.rs`).

use ndarray::Array2;
use snapseg_core::Point2;
use std::collections::HashMap;

/// A directed cell-edge midpoint. Carries enough information to (a) snap to
/// a unique integer key for graph dedup, and (b) materialise as an `(x, y)`
/// half-pixel point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EdgeKey {
    /// `true` for horizontal cell edges (midpoint at `x + 0.5, y`),
    /// `false` for vertical cell edges (midpoint at `x, y + 0.5`).
    horizontal: bool,
    /// Integer pixel row (for horizontal edges) or pixel row containing the
    /// edge midpoint (for vertical edges).
    row: i32,
    /// Integer pixel column (for vertical edges) or pixel column containing
    /// the edge midpoint (for horizontal edges).
    col: i32,
}

impl EdgeKey {
    fn to_point(self) -> Point2 {
        if self.horizontal {
            Point2::new(self.col as f32 + 0.5, self.row as f32)
        } else {
            Point2::new(self.col as f32, self.row as f32 + 0.5)
        }
    }
}

/// Extract closed polylines from `mask`. Returns at most as many polylines
/// as there are disconnected foreground/background interfaces. Empty mask
/// or all-foreground mask returns an empty vector.
pub(crate) fn marching_squares(mask: &Array2<bool>) -> Vec<Vec<Point2>> {
    let (h, w) = mask.dim();
    if h < 2 || w < 2 {
        return Vec::new();
    }

    let segments = collect_segments(mask);
    if segments.is_empty() {
        return Vec::new();
    }
    walk_polylines(&segments)
}

/// Emit one or two undirected segments per 2x2 cell.
fn collect_segments(mask: &Array2<bool>) -> Vec<(EdgeKey, EdgeKey)> {
    let (h, w) = mask.dim();
    let mut segments: Vec<(EdgeKey, EdgeKey)> = Vec::new();
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let tl = mask[(y, x)];
            let tr = mask[(y, x + 1)];
            let br = mask[(y + 1, x + 1)];
            let bl = mask[(y + 1, x)];
            let code = (tl as u8) << 3 | (tr as u8) << 2 | (br as u8) << 1 | (bl as u8);
            let top = EdgeKey {
                horizontal: true,
                row: y as i32,
                col: x as i32,
            };
            let bot = EdgeKey {
                horizontal: true,
                row: (y + 1) as i32,
                col: x as i32,
            };
            let left = EdgeKey {
                horizontal: false,
                row: y as i32,
                col: x as i32,
            };
            let right = EdgeKey {
                horizontal: false,
                row: y as i32,
                col: (x + 1) as i32,
            };
            push_segments(code, top, right, bot, left, &mut segments);
        }
    }
    segments
}

/// 16-case dispatch. Saddle resolution is fixed: corners avg `>= 0.5` (true
/// for boolean masks) connects inside through the cell.
///
/// `// why this is long:` 16 mutually-exclusive case arms with one or two
/// `push` calls each — splitting them harms readability more than it helps.
fn push_segments(
    code: u8,
    top: EdgeKey,
    right: EdgeKey,
    bot: EdgeKey,
    left: EdgeKey,
    out: &mut Vec<(EdgeKey, EdgeKey)>,
) {
    match code {
        0b0000 | 0b1111 => {}
        0b0001 => out.push((left, bot)),
        0b0010 => out.push((bot, right)),
        0b0011 => out.push((left, right)),
        0b0100 => out.push((top, right)),
        0b0101 => {
            // Saddle: TR + BL inside. Connect inside (>= 0.5) → segments
            // are L-T and B-R.
            out.push((left, top));
            out.push((bot, right));
        }
        0b0110 => out.push((top, bot)),
        0b0111 => out.push((left, top)),
        0b1000 => out.push((left, top)),
        0b1001 => out.push((top, bot)),
        0b1010 => {
            // Saddle: TL + BR inside. Connect inside through cell → L-B
            // and T-R.
            out.push((left, bot));
            out.push((top, right));
        }
        0b1011 => out.push((top, right)),
        0b1100 => out.push((left, right)),
        0b1101 => out.push((bot, right)),
        0b1110 => out.push((left, bot)),
        _ => unreachable!("4-bit code"),
    }
}

/// Walk an undirected segment graph into closed polylines.
fn walk_polylines(segments: &[(EdgeKey, EdgeKey)]) -> Vec<Vec<Point2>> {
    let mut adj: HashMap<EdgeKey, Vec<EdgeKey>> = HashMap::new();
    for &(a, b) in segments {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }

    let mut polylines: Vec<Vec<Point2>> = Vec::new();
    while let Some(&start) = adj.keys().find(|k| !adj[*k].is_empty()) {
        let mut chain: Vec<EdgeKey> = Vec::new();
        let mut cur = start;
        chain.push(cur);
        loop {
            let next_opt = pop_neighbour(&mut adj, cur);
            match next_opt {
                Some(next) => {
                    // Remove the reciprocal half-edge too.
                    remove_one(&mut adj, next, cur);
                    if next == start {
                        // Closed loop; emit start as last vertex to mark
                        // closure.
                        chain.push(next);
                        break;
                    }
                    chain.push(next);
                    cur = next;
                }
                None => break,
            }
        }
        if chain.len() >= 3 {
            let pts: Vec<Point2> = chain.into_iter().map(EdgeKey::to_point).collect();
            polylines.push(pts);
        }
    }
    polylines
}

fn pop_neighbour(adj: &mut HashMap<EdgeKey, Vec<EdgeKey>>, k: EdgeKey) -> Option<EdgeKey> {
    let list = adj.get_mut(&k)?;
    list.pop()
}

fn remove_one(adj: &mut HashMap<EdgeKey, Vec<EdgeKey>>, k: EdgeKey, target: EdgeKey) {
    if let Some(list) = adj.get_mut(&k) {
        if let Some(pos) = list.iter().position(|&x| x == target) {
            list.swap_remove(pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn empty_mask_yields_no_polylines() {
        let mask = Array2::<bool>::from_elem((10, 10), false);
        assert!(marching_squares(&mask).is_empty());
    }

    #[test]
    fn full_mask_yields_no_polylines() {
        let mask = Array2::<bool>::from_elem((10, 10), true);
        assert!(marching_squares(&mask).is_empty());
    }

    #[test]
    fn small_square_one_closed_polyline() {
        let mut mask = Array2::<bool>::from_elem((8, 8), false);
        for y in 2..6 {
            for x in 2..6 {
                mask[(y, x)] = true;
            }
        }
        let lines = marching_squares(&mask);
        assert_eq!(lines.len(), 1);
        let line = &lines[0];
        // Closed: first == last.
        let first = line[0];
        let last = line[line.len() - 1];
        assert!((first.x - last.x).abs() < 1e-6);
        assert!((first.y - last.y).abs() < 1e-6);
        // 4x4 square → 4 sides of 4 cells each = 16 segments → 17 entries
        // (closure repeats start) but exact count depends on walking; assert
        // a sane lower bound.
        assert!(line.len() >= 16);
    }
}
