{
  pkgs,
  lib,
  stdenv,
  ...
}:
let
  pkgs-unfree = import pkgs.path {
    inherit (pkgs) system;
    config.allowUnfree = true;
  };
  pythonPackages = pkgs-unfree.python313Packages;
in
pkgs-unfree.mkShell {
  buildInputs = [
    pythonPackages.python
    pythonPackages.venvShellHook
    pkgs-unfree.autoPatchelfHook
    pythonPackages.onnxruntime
    pkgs-unfree.onnxruntime
    pythonPackages.datasets
    pythonPackages.torchvision
    pythonPackages.peft
    pythonPackages.evaluate
    pythonPackages.jiwer
    pythonPackages.accelerate
    pythonPackages.torch
    pythonPackages.tensorboard
    pythonPackages.scikit-learn
    pythonPackages.pillow
    pkgs-unfree.cudaPackages.cudatoolkit

  ];
  venvDir = "./.venv";
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    autoPatchelf ./.venv
  '';
  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH=${pkgs-unfree.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${
      lib.makeLibraryPath [
        stdenv.cc.cc
      ]
    }:$LD_LIBRARY_PATH
    echo "python -c 'import torch; print(f\"CUDA disponível: {torch.cuda.is_available()}\")'"
  '';
}
