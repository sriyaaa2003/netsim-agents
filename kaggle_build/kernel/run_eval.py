import sys, glob, os
root = glob.glob("/kaggle/input/**/netsimagents", recursive=True)
sys.path.insert(0, os.path.dirname(root[0]))
from netsimagents.evaluate import main
main(["--hf-model", "Qwen/Qwen2.5-7B-Instruct", "--out", "/kaggle/working/results"])
