# Formula/airml.rb — Homebrew formula for airML
#
# NOTE: The `sha256 "REPLACE_AT_RELEASE_TIME"` placeholders below are
# automatically filled in by `.github/workflows/release.yml` when a version
# tag is pushed. Do not edit the sha256 values by hand.
#
# To add this tap:
#   brew tap airml/airml https://github.com/airml/homebrew-airml
#   brew install airml
#
# Or install directly from this formula file for testing:
#   brew install --formula Formula/airml.rb

class Airml < Formula
  desc "Run any ONNX model on Apple Silicon as a single binary"
  homepage "https://github.com/airml/airml"
  version "0.2.0"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/airml/airml/releases/download/v0.2.0/airml-macos-aarch64-v0.2.0.tar.gz"
      sha256 "REPLACE_AT_RELEASE_TIME"
    else
      url "https://github.com/airml/airml/releases/download/v0.2.0/airml-macos-x86_64-v0.2.0.tar.gz"
      sha256 "REPLACE_AT_RELEASE_TIME"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/airml/airml/releases/download/v0.2.0/airml-linux-aarch64-v0.2.0.tar.gz"
      sha256 "REPLACE_AT_RELEASE_TIME"
    else
      url "https://github.com/airml/airml/releases/download/v0.2.0/airml-linux-x86_64-v0.2.0.tar.gz"
      sha256 "REPLACE_AT_RELEASE_TIME"
    end
  end

  def install
    bin.install "airml"
  end

  test do
    system "#{bin}/airml", "--version"
  end
end
