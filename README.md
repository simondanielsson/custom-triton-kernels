# Custom GPU kernel implementations in Triton for DL workflows

Project is structured as follows:

```bash
.
├── src
│   └── kernels
│       ├── __init__.py
│       ├── dropout.py
│       └── softmax.py
│       └── ...
├── tests
│   ├── __init__.py
│   └── dropout_test.py
│   └── softmax_test.py
│   └── ...

```

Currently supported kernels:

- Softmax
- Dropout
- Vector addition

More to be implemented. Coming up:

- [] Matrix addition
- [] Block-based fused softmax
- [] Layer norm
- [] Matrix multiplication
- [] Fused attention

## Testing kernel correctness

To run tests, simply run

```bash
make test
```

## Prepare your environment

```bash
python3 -m venv venv
source venv/bin/activate
make install-dev
```
