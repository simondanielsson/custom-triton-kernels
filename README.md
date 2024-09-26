# Custom GPU kernel implementations in Triton for DL workflows

Project is structured as follows:

```bash
.
├── kernels
│   ├── dropout.py
│   └── softmax.py
│   └── ...
├── tests
│   ├── dropout.py
│   └── softmax.py
│   └── ...

```

Currently supported kernels:

- Softmax
- Dropout

More to be implemented.

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
