# Overview
Here we provide an implementation of LMON (Latent Motif Oriented Network) in PyTorch. The repository is organised as follows:
- `data/` contains the necessary dataset files for Cora;
- `new_data/` contains the dataset (cornell, film, texas, wisconsin);
- `models/` contains the implementation of the proposed model (LMON);
- `layers/` contains the implementation of a GCN layer (`gcn.py`),  a LSTM layer (lstm.py), the averaging readout (`readout.py`), and the bilinear discriminator (`discriminator.py`);
- `utils/` contains the necessary processing subroutines (`process.py`).

Finally, `execute.py` puts all of the above together and may be used to execute a full training run on dataset.

## License
MIT
