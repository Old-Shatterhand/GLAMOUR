import sys
import os
import random
from pathlib import Path
from typing import Literal

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from utils.load_networkx import networkx_feat
from utils.macro_dataset import MacroDataset
from utils.macro_supervised import MacroSupervised

FEAT = 'fp'
FP_RADIUS_MON = 3
FP_BITS_MON = 128
FP_RADIUS_BOND = 3
FP_BITS_BOND = 16
SEED = 42
NORM = "qt"


def train(base: Path, label_name: str, model_name: Literal["GCN", "GAT", "Weave", "MPNN", "AttentiveFP"]):
    if "," in label_name:
        label_name = label_name.split(",")
    mono_file = base / "monos.txt"
    bond_file = base / "bonds.txt"
    graph_path = base / "graphs"
    if "taxonomy" in str(base):
        df_path = base / "multilabel.txt"
    else:
        df_path = base / "classificaion.txt"

    graphs = networkx_feat(
        TXT_DATA_PATH=graph_path,
        MON_SMILES=mono_file,
        BOND_SMILES=bond_file,
        FEAT=FEAT,
        FP_RADIUS_MON=FP_RADIUS_MON,
        FP_RADIUS_BOND=FP_RADIUS_BOND,
        FP_BITS_MON=FP_BITS_MON,
        FP_BITS_BOND=FP_BITS_BOND
    )

    dataset = MacroDataset(
        DF_PATH=df_path,
        SEED=SEED,
        TASK="classification",
        LABELNAME=label_name,
        MODEL=model_name,
        NX_GRAPHS=graphs,
        NORM=NORM
    )

    random.seed(SEED)

    vs = [int(x.stem.split("_")[-1]) for x in filter(lambda x: "version_" in str(x), base.iterdir())]
    if vs is not None and len(vs) > 0:
        version = base / f"version_{max(vs) + 1}"
    else:
        version = base / "version_0"

    macro_supervised = MacroSupervised(
        MacroDataset=dataset,
        MON_SMILES=mono_file,
        BOND_SMILES=bond_file,
        FEAT=FEAT,
        FP_BITS_MON=FP_BITS_MON,
        FP_BITS_BOND=FP_BITS_BOND,
        SEED=SEED,
        MODEL=model_name,
        SPLIT="0.6,0.2,0.2",  # going to be ignored
        NUM_EPOCHS=100,
        NUM_WORKERS=16,
        CUSTOM_PARAMS={},
        MODEL_PATH=str(version),
        SAVE_MODEL=True,
        SAVE_OPT=True,
        SAVE_CONFIG=True,
    )

    macro_supervised.main()


if __name__ == '__main__':
    base, label_name, model_name = sys.argv[1:4]
    train(Path(base), label_name, model_name)
