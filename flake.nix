# -----------------------------------------------------------------------------
# Top-level flake configuration for the Rigel repository developer environment.
# -----------------------------------------------------------------------------
# Mirrors the pattern used in projects/wtgen/ so that both `nix develop` and `devenv`
# share the exact same shell configuration defined in devenv.nix.
# -----------------------------------------------------------------------------
{
  inputs = {
    # Follow the same rolling nixpkgs channel maintained by devenv to minimize
    # manual maintenance and stay aligned with devenv's options.
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling";

    # Common set of supported target systems (Linux/macOS, x86_64/aarch64).
    systems.url = "github:nix-systems/default";

    # devenv itself, following nixpkgs so that updates stay in sync.
    devenv.url = "github:cachix/devenv";
    devenv.inputs.nixpkgs.follows = "nixpkgs";

    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  # Reuse the public binary cache provided by devenv and allow access to PWD so
  # devenv.nix can pick up the repository root without requiring --impure.
  nixConfig = {
    extra-trusted-public-keys =
      "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
    extra-experimental-features = "configurable-impure-env";
    impure-env = "PWD";
  };

  outputs = { self, nixpkgs, devenv, systems, rust-overlay, ... } @ inputs:
    let
      # Helper to instantiate attributes for each supported CPU/OS pair.
      forEachSystem = nixpkgs.lib.genAttrs (import systems);

      # Share the exact same devenv shell between `nix develop` and
      # `devenv shell`, keeping configuration centralized in devenv.nix.
      mkDevShell = system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
        in
        devenv.lib.mkShell {
          inherit inputs pkgs;
          modules = [ ./devenv.nix ];
        };
    in
    {
      devShells = forEachSystem (system: { default = mkDevShell system; });
      devenv.shells = forEachSystem (system: { default = mkDevShell system; });
    };
}
