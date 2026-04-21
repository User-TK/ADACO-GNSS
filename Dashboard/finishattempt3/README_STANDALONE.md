# Standalone ADACO-GNSS demo

This version removes FastAPI/Uvicorn entirely.

## Minimal layout

```text
standalone_demo.py
cnn_best.pt
spectrum_cnn.py
demodata/
  0.json
  1.json
  ...
```

Or, if you want to keep your original project layout:

```bash
python standalone_demo.py --model models/cnn_best.pt --data demodata
```

## Run

```bash
python standalone_demo.py
```

Then open:

```text
http://127.0.0.1:8000/
```

## Dependencies

The web server itself uses Python's standard library, but the model still requires:

```bash
pip install torch numpy scikit-learn
```

`scikit-learn` is needed if `cnn_best.pt` contains a saved `StandardScaler`, which your current `predictor.py` says it does.

## Important note about the .pt file

Your current loading method uses:

```python
self.model = HybridGNSSCNN(in_channels=2, scalar_dim=30)
self.model.load_state_dict(ckpt["model_state"])
```

That means `cnn_best.pt` does **not** fully describe the model architecture by itself. The standalone file still needs access to `HybridGNSSCNN`, usually by keeping `spectrum_cnn.py` next to `standalone_demo.py`.

If you want a truly one-file Python demo plus `cnn_best.pt`, paste the full `HybridGNSSCNN` class into `standalone_demo.py` and remove this line:

```python
from spectrum_cnn import HybridGNSSCNN
```
