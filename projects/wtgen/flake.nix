# -----------------------------------------------------------------------------
# Top-level flake configuration for the wtgen developer environment.
# -----------------------------------------------------------------------------
# This flake is consumed both by `nix develop` and by `devenv` itself.
# Everything else (languages, packages, shell hooks, scripts, etc.) lives in
# `devenv.nix` so that developers only need to learn a single file when tweaking
# the environment.
# -----------------------------------------------------------------------------
{
  inputs = {
    # We follow the rolling nixpkgs channel maintained by the devenv project.
    # It is kept in sync with devenv's options and dramatically reduces the
    # amount of manual maintenance we need to do.
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling";

    # Common set of target systems we want to support. This lets the flake
    # produce shells for both Linux (x86_64/aarch64) and macOS (x86_64/aarch64).
    systems.url = "github:nix-systems/default";

    # devenv itself. We keep the follow relationship below so we do not have to
    # repeat the nixpkgs pin when devenv updates.
    devenv.url = "github:cachix/devenv";
    devenv.inputs.nixpkgs.follows = "nixpkgs";

    # Extra repository providing pre-built CPython interpreters that can be used
    # with the `languages.python.version` option.
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
    nixpkgs-python.inputs = { nixpkgs.follows = "nixpkgs"; };

    nix2container.url = "github:nlewo/nix2container";
    nix2container.inputs.nixpkgs.follows = "nixpkgs";

    mk-shell-bin.url = "github:rrbutani/nix-mk-shell-bin";
  };

  # Use the public binary cache provided by devenv so developers do not need to
  # build large packages from source on first install.
  nixConfig = {
    extra-trusted-public-keys =
      "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
    extra-experimental-features = "configurable-impure-env";
    impure-env = "PWD DEVENV_CONTAINER_VERSION";
  };

  outputs = { self, nixpkgs, devenv, systems, nix2container, mk-shell-bin, ... } @ inputs:
    let
      # Helper to instantiate attributes for each supported CPU/OS pair.
      forEachSystem = nixpkgs.lib.genAttrs (import systems);

      # Convenience helper so we can share the exact same devenv shell between
      # `nix develop` and `devenv shell`.
      mkDevShell = system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        devenv.lib.mkShell {
          inherit inputs pkgs;
          modules = [
            # All of the interesting, heavily documented configuration lives in
            # this module file.
            ./devenv.nix
          ];
        };
    in
    {
      # `nix develop` users land here.
      devShells = forEachSystem (system: { default = mkDevShell system; });

      # `devenv shell` users land here. Keeping both in sync ensures the same
      # experience regardless of preferred entrypoint.
      devenv.shells = forEachSystem (system: { default = mkDevShell system; });
    };
}
