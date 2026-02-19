import os
import subprocess
import shutil
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

csrc = Path(__file__).parent / "opensplat_metal" / "csrc"

# Compile Metal shader to metallib at build time
def compile_metallib():
    metal_src = csrc / "gsplat_metal.metal"
    air_file = csrc / "gsplat_metal.air"
    metallib_file = csrc / "default.metallib"
    subprocess.check_call([
        "xcrun", "-sdk", "macosx", "metal",
        "-c", str(metal_src),
        "-o", str(air_file),
    ])
    subprocess.check_call([
        "xcrun", "-sdk", "macosx", "metallib",
        str(air_file),
        "-o", str(metallib_file),
    ])
    air_file.unlink(missing_ok=True)
    return metallib_file

metallib_path = compile_metallib()

class CustomBuildExtension(BuildExtension):
    def run(self):
        super().run()
        # Copy metallib next to the built .so
        for ext in self.extensions:
            ext_path = self.get_ext_fullpath(ext.name)
            dest_dir = Path(ext_path).parent
            dest = dest_dir / "default.metallib"
            shutil.copy2(str(metallib_path), str(dest))
            print(f"Copied default.metallib -> {dest}")

setup(
    name="opensplat-metal",
    ext_modules=[
        CppExtension(
            name="opensplat_metal._C",
            sources=[
                os.path.join("opensplat_metal", "csrc", "ext.mm"),
                os.path.join("opensplat_metal", "csrc", "gsplat_metal.mm"),
            ],
            extra_compile_args={
                "cxx": ["-std=c++17"],
            },
            extra_link_args=[
                "-framework", "Metal",
                "-framework", "MetalKit",
                "-framework", "Foundation",
            ],
        ),
    ],
    cmdclass={"build_ext": CustomBuildExtension},
    packages=["opensplat_metal"],
)
