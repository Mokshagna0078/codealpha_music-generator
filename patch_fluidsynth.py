"""Patch fluidsynth.py to handle missing DLL directory without crashing."""
import os

path = os.path.expanduser(
    "~/AppData/Local/Programs/Python/Python311/Lib/site-packages/fluidsynth.py"
)

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the lines to patch
# Lines 46-50 (0-indexed): os.add_dll_directory calls
for i, line in enumerate(lines):
    if 'os.add_dll_directory' in line and 'os.getcwd()' in line:
        # Wrap this in try/except
        indent = "    "
        lines[i] = f"{indent}try:\n{indent}    os.add_dll_directory(os.getcwd())\n{indent}except (FileNotFoundError, OSError):\n{indent}    pass\n"
        print(f"Patched line {i+1}: os.add_dll_directory(os.getcwd())")
    elif "os.add_dll_directory" in line and "fluidsynth" in line:
        indent = "    "
        lines[i] = f"{indent}try:\n{indent}    os.add_dll_directory('C:\\\\tools\\\\fluidsynth\\\\bin')\n{indent}except (FileNotFoundError, OSError):\n{indent}    pass\n"
        print(f"Patched line {i+1}: os.add_dll_directory(fluidsynth)")
    elif "os.environ['PATH']" in line and "fluidsynth" in line:
        indent = "    "
        lines[i] = f"{indent}try:\n{indent}    os.environ['PATH'] += ';C:\\\\tools\\\\fluidsynth\\\\bin'\n{indent}except Exception:\n{indent}    pass\n"
        print(f"Patched line {i+1}: PATH += fluidsynth")

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Patch complete!")
