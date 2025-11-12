from __future__ import annotations
import json
from pymatgen.core import Structure
from dgl.data.utils import split_dataset
from matgl.ext.pymatgen import Structure2Graph
import warnings
from functools import partial
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping
import numpy as np
from pytorch_lightning.loggers import CSVLogger
from diep.config import DEFAULT_ELEMENTS
from diep.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes
from diep.models import DIEP
from diep.utils.training import PotentialLightningModule
import torch
import random
from ase.stress import voigt_6_to_full_3x3_stress
warnings.simplefilter("ignore")
import collections
from tqdm import tqdm
from diep.utils.training import xavier_init
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from lightning.pytorch.callbacks import ModelCheckpoint

DIR = f"logs"
EPOCHS = 200
ACCELERATOR = "gpu"
DEVICES = 2
WORKERS = 0
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

with open("MatPES-R2SCAN-atoms.json", "r") as f:
    isolated_energies_pbe = json.load(f)
isolated_energies_pbe = {d["elements"][0]: d["energy"] for d in isolated_energies_pbe}

with open("MatPES-R2SCAN-2025.1.json", "r") as f:
    data = json.load(f)

structures = []
labels = collections.defaultdict(list)
for d in tqdm(data):
    structures.append(Structure.from_dict(d["structure"]))
    labels["energies"].append(d["energy"])
    labels["forces"].append(d["forces"])
    labels["stresses"].append(
        (voigt_6_to_full_3x3_stress(np.array(d["stress"])) * -0.1).tolist()
    )

element_types = DEFAULT_ELEMENTS
cutoff = 5.0
threebody_cutoff = 4.0

converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
dataset = MGLDataset(
    structures=structures,
    include_line_graph=True,
    threebody_cutoff=threebody_cutoff,
    converter=converter,
    labels=labels,
    save_cache=False,
)

training_set, validation_set, test_set = split_dataset(
    dataset, frac_list=[0.9, 0.05, 0.05], random_state=42, shuffle=True
)
collate_fn = partial(collate_fn_pes, include_line_graph=True, include_stress=True)
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=training_set,
    val_data=validation_set,
    test_data=test_set,
    collate_fn=collate_fn,
    batch_size=32,
    num_workers=WORKERS,
)
model = DIEP(
    element_types=element_types,
    is_intensive=False,
    units=128,
    readout_type="weighted_atom",
    grid_half_length=3.0,
    base_spacing=0.5,
    gaussian_sigma=1.0,
    integral_mode="grid",
    softening_epsilon=0.5,
    use_effective_charge=True,
    use_smooth=True
)

train_graphs = []
energies = []
forces = []

for _g, _lat, _line_graph, _attrs, lbs in training_set:
    forces.append(lbs["forces"])
forces = torch.concatenate(forces)
rms_forces = torch.sqrt(torch.mean(torch.sum(forces**2, dim=1)))

xavier_init(model)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=1.0e-3, weight_decay=1.0e-5, amsgrad=True
)
scheduler = CosineAnnealingLR(optimizer, T_max=1000 * 10, eta_min=1.0e-2 * 1.0e-3)
energies_offsets = np.array(
    [isolated_energies_pbe[element] for element in DEFAULT_ELEMENTS]
)

lit_model = PotentialLightningModule(
    model=model,
    element_refs=energies_offsets,
    data_std=rms_forces,
    optimizer=optimizer,
    scheduler=scheduler,
    loss="l1_loss",
    stress_weight=0.1,
    include_line_graph=True,
)

path = os.getcwd()
logger = CSVLogger(save_dir=path, name=DIR)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_Total_Loss",
    mode="min",
    filename="{epoch:04d}-best_model",
)

trainer = pl.Trainer(
    logger=logger,
    callbacks=[
        EarlyStopping(monitor="val_Total_Loss", mode="min", patience=200),
        checkpoint_callback,
    ],
    max_epochs=EPOCHS,
    accelerator=ACCELERATOR,
    gradient_clip_val=2.0,
    accumulate_grad_batches=4,
    devices=DEVICES,
    inference_mode=False,
)
trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
model_export_path = os.path.join(path, DIR, "trained_model")
os.makedirs(model_export_path, exist_ok=True)
lit_model.model.save(model_export_path)

trainer.test(dataloaders=test_loader)
