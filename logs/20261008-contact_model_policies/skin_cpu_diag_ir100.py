import sys; from pathlib import Path
sys.argv = [sys.argv[0]]
exec(open(Path(__file__).with_name("skin_cpu_diag.py")).read().split("for name, ir in")[0])
for name, ir in (("pads_ir10", 100), ("skin_ir1000", 100), ("pads_ir10", 300), ("skin_ir1000", 300)):
    print(name, "impratio", ir, run(scenes[name], ir), flush=True)
