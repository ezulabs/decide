//! Compiles `swift/ane.swift` into a static library and links it, with the
//! Swift runtime from the OS (`/usr/lib/swift`) and the frameworks it uses.
//! A missing `swiftc` fails the build: the ANE path has no fallback here (the
//! engine's switch is what turns it off).
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("swift/ane.swift");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", src.display());

    let lib = out_dir.join("libdecide_ane_swift.a");
    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "swiftc", "-O", "-parse-as-library", "-emit-library", "-static"])
        .args(["-module-name", "decide_ane_swift"])
        .arg(&src)
        .arg("-o")
        .arg(&lib)
        .output()
        .expect("cannot run `xcrun swiftc`: install Xcode's command line tools");
    assert!(
        output.status.success(),
        "swiftc FAILED for {} ({}):\n{}",
        src.display(),
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let sdk = Command::new("xcrun").args(["-sdk", "macosx", "--show-sdk-path"]).output().expect("xcrun --show-sdk-path");
    let sdk = String::from_utf8_lossy(&sdk.stdout).trim().to_string();
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=decide_ane_swift");
    println!("cargo:rustc-link-search=native={sdk}/usr/lib/swift");
    println!("cargo:rustc-link-search=native=/usr/lib/swift");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    // The Swift runtime and overlays the shim uses (the OS ships them under
    // /usr/lib/swift; the SDK has the .tbd stubs). Only the ones present in
    // this SDK are named; the object's own autolink entries cover the rest.
    for lib in ["swiftCore", "swiftFoundation", "swiftDispatch", "swiftObjectiveC", "swiftCoreFoundation", "swiftDarwin",
                "swiftMetal", "swiftCoreML", "swiftQuartzCore", "swiftos", "swiftXPC", "swiftIOKit",
                "swiftCoreGraphics", "swiftCoreImage", "swiftUniformTypeIdentifiers", "swiftDataDetection", "swiftsimd"] {
        if PathBuf::from(format!("{sdk}/usr/lib/swift/lib{lib}.tbd")).exists() {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
    for fw in ["CoreML", "Metal", "Foundation", "QuartzCore", "CoreVideo", "IOSurface"] {
        println!("cargo:rustc-link-lib=framework={fw}");
    }
}
