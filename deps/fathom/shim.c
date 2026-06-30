/* Thin shim exposing linkable symbols for the Rust FFI.
 *
 * Fathom's tb_probe_wdl / tb_probe_root are `static inline` in tbprobe.h (they
 * do the castling/rule50 pre-checks then call the *_impl symbols), so they are
 * not linkable directly. These wrappers call them and ARE real symbols. */
#include <stddef.h>
#include <stdint.h>

#include "tbprobe.h"

int ek_tb_init(const char *path) { return tb_init(path) ? 1 : 0; }

void ek_tb_free(void) { tb_free(); }

unsigned ek_tb_largest(void) { return TB_LARGEST; }

unsigned ek_tb_probe_wdl(uint64_t white, uint64_t black, uint64_t kings,
                         uint64_t queens, uint64_t rooks, uint64_t bishops,
                         uint64_t knights, uint64_t pawns, unsigned rule50,
                         unsigned castling, unsigned ep, int turn) {
  return tb_probe_wdl(white, black, kings, queens, rooks, bishops, knights,
                      pawns, rule50, castling, ep, turn != 0);
}

unsigned ek_tb_probe_root(uint64_t white, uint64_t black, uint64_t kings,
                          uint64_t queens, uint64_t rooks, uint64_t bishops,
                          uint64_t knights, uint64_t pawns, unsigned rule50,
                          unsigned castling, unsigned ep, int turn) {
  return tb_probe_root(white, black, kings, queens, rooks, bishops, knights,
                       pawns, rule50, castling, ep, turn != 0, NULL);
}
