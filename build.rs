// Compiles the vendored Fathom Syzygy prober (deps/fathom/) + our shim into a
// static lib linked by the FFI in src/syzygy.rs.
fn main() {
    println!("cargo:rerun-if-changed=deps/fathom/tbprobe.c");
    println!("cargo:rerun-if-changed=deps/fathom/shim.c");
    println!("cargo:rerun-if-changed=deps/fathom/tbprobe.h");
    println!("cargo:rerun-if-changed=deps/fathom/tbconfig.h");

    cc::Build::new()
        .file("deps/fathom/tbprobe.c")
        .file("deps/fathom/shim.c")
        .include("deps/fathom")
        .opt_level(3)
        .warnings(false)
        .compile("fathom");
}
