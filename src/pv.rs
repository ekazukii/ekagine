use chess::ChessMove;

/// Maximum supported ply depth for the principal variation table.
///
/// This should comfortably exceed the deepest search depth we plan to reach.
pub const PV_MAX_PLY: usize = 256;

/// A fixed-size principal variation table indexed by ply.
///
/// Each row stores the best move sequence (PV) starting at that ply. When a
/// new best move is found at ply *p*, we record it in the diagonal entry
/// `table[p][p]` and copy the continuation from the child ply so that
/// `table[0]` always holds the current principal variation from the root.
#[derive(Clone)]
pub struct PVTable {
    table: Vec<[Option<ChessMove>; PV_MAX_PLY]>,
}

impl PVTable {
    /// Create a new empty PV table.
    pub fn new() -> Self {
        Self {
            table: vec![[None; PV_MAX_PLY]; PV_MAX_PLY],
        }
    }

    /// Remove all stored moves.
    pub fn clear(&mut self) {
        for row in &mut self.table {
            *row = [None; PV_MAX_PLY];
        }
    }

    /// Clear the PV line stored at the given ply (from that ply onward).
    pub fn clear_line_from(&mut self, ply: usize) {
        if ply >= PV_MAX_PLY {
            return;
        }
        let row = &mut self.table[ply];
        for col in ply..PV_MAX_PLY {
            row[col] = None;
        }
    }

    /// Record a new best move at `ply` and copy the child's continuation.
    pub fn update_line(&mut self, ply: usize, mv: ChessMove) {
        if ply >= PV_MAX_PLY {
            return;
        }

        let (current_prefix, rest) = self.table.split_at_mut(ply + 1);
        let current_row = &mut current_prefix[ply];
        current_row[ply] = Some(mv);

        if let Some(next_row) = rest.first() {
            for col in (ply + 1)..PV_MAX_PLY {
                current_row[col] = next_row[col];
            }
        } else {
            for col in (ply + 1)..PV_MAX_PLY {
                current_row[col] = None;
            }
        }
    }

    /// Return the stored PV move at the specified ply, if any.
    pub fn best_move_at(&self, ply: usize) -> Option<ChessMove> {
        if ply >= PV_MAX_PLY {
            return None;
        }
        self.table[ply][ply]
    }

    /// Return the principal variation starting from the root.
    pub fn principal_variation(&self) -> Vec<ChessMove> {
        self.line_from(0)
    }

    /// Return the PV sequence beginning at the provided ply.
    pub fn line_from(&self, ply: usize) -> Vec<ChessMove> {
        let mut line = Vec::new();
        if ply >= PV_MAX_PLY {
            return line;
        }
        for col in ply..PV_MAX_PLY {
            if let Some(mv) = self.table[ply][col] {
                line.push(mv);
            } else {
                break;
            }
        }
        line
    }
}

impl Default for PVTable {
    fn default() -> Self {
        Self::new()
    }
}
