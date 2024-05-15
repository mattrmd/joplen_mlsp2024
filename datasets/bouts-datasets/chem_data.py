import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import numpy as np
import subprocess

# get path of current file
CURRENT_PATH = Path(__file__).parent.resolve()


def pull_raw_data():
    """Pull raw data from remote storage using dvc pull."""
    dir_path = CURRENT_PATH / "raw_data"

    for file in dir_path.iterdir():
        if file.suffix != ".dvc":
            continue

        file = file.with_suffix("")
        subprocess.run(["dvc", "pull", str(file)], check=True)


class ChemData(ABC):
    def pull_data(self):
        return self.get_data()

    def __init__(
        self,
        filename: str,
        y_col: str,
        drop_cols: list,
        name: str,
        index_col: Union[int, None] = None,
        drop_const: bool = True,
        drop_corr: bool = True,
    ):
        self.data_dir = CURRENT_PATH / "preprocessed_data"

        file_path = self.data_dir / filename

        # download the file from remote if it doesn't exist
        if not file_path.exists():
            subprocess.run(["dvc", "pull", str(file_path)], check=True)

        df = pd.read_csv(file_path, index_col=index_col)

        self.y = df.pop(y_col)
        self.x = df.drop(columns=drop_cols)

        if drop_const:
            self.x = self.x.loc[:, self.x.std() != 0]

        if drop_corr:
            C = np.corrcoef(self.x.values, rowvar=False)

            # Initialize a list to keep track of columns to drop
            cols_to_drop = []

            # Loop to identify perfectly correlated columns
            n_cols = C.shape[0]
            for i in range(n_cols):
                for j in range(i + 1, n_cols):
                    if abs(C[i, j]) == 1.0:
                        cols_to_drop.append(self.x.columns[j])

            # Make the list of columns to drop unique
            cols_to_drop = list(set(cols_to_drop))

            # Drop the identified columns
            self.x.drop(columns=cols_to_drop, inplace=True)

        self.features = self.x.columns
        self.label = y_col
        self.name = name
        self.index = df.index

    def get_data(
        self,
        as_numpy: bool = True,
    ):
        x, y = self.x, self.y
        index = self.index

        if as_numpy:
            x, y = x.to_numpy(), y.to_numpy()
            y = y.reshape(-1, 1)
            index = index.to_numpy()

        return x, y, index


class NPLogP(ChemData):
    def __init__(self, **kwargs):
        super().__init__(
            filename="nanoparticle_logp.csv",
            y_col="logp",
            drop_cols=[],
            name="NP LogP",
            index_col=0,
            **kwargs,
        )


class NPZetaP(ChemData):
    def __init__(self, **kwargs):
        super().__init__(
            filename="nanoparticle_zp.csv",
            y_col="zeta_potential",
            drop_cols=[],
            name="NP ZetaP",
            index_col=0,
            **kwargs,
        )


class ProtSol(ChemData):
    def __init__(self, **kwargs):
        super().__init__(
            filename="new_protein_sol.csv",
            y_col="solubility",
            drop_cols=[],
            name="Protein Sol",
            index_col=0,
            **kwargs,
        )


class MolLogP(ChemData):
    def __init__(self, **kwargs):
        super().__init__(
            filename="small_molecules_logp.csv",
            y_col="logp",
            drop_cols=["smiles"],
            name="Mol LogP",
            index_col=0,
            **kwargs,
        )


class MolHenry(ChemData):
    def __init__(self, **kwargs):
        super().__init__(
            filename="physprop_henry.csv",
            y_col="henry",
            drop_cols=["smiles"],
            name="Mol Henry",
            index_col=0,
            **kwargs,
        )


class MolBoil(ChemData):
    def __init__(self, **kwargs):
        super().__init__(
            filename="physprop_boiling.csv",
            y_col="boiling",
            drop_cols=["smiles"],
            name="Mol Boil",
            index_col=0,
            **kwargs,
        )


class MolMelt(ChemData):
    def __init__(self, **kwargs):
        super().__init__(
            filename="physprop_melting.csv",
            y_col="melting",
            drop_cols=["smiles"],
            name="Mol Melt",
            index_col=0,
            **kwargs,
        )


if __name__ == "__main__":
    from tqdm import tqdm

    path = Path(__file__).parent.resolve()

    datasets = [
        NPLogP,
        NPZetaP,
        ProtSol,
        MolLogP,
        MolHenry,
        MolBoil,
        MolMelt,
    ]

    itr = tqdm(datasets)
    for ds in itr:
        itr.set_description(ds.__name__)
        ds(path)
