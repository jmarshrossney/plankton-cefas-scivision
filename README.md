# resnet50-cefas

This is refactored version of [this project](https://github.com/alan-turing-institute/plankton-cefas-scivision) that has been stripped down to the bare minimum and updated so that it can be used into a Python >3.11 environment.

## Usage

You can install this repository as a package using

```sh
python -m pip install git+https://github.com/jmarshrossney/resnet50-cefas
```

and then load the model into your script as follows:

```python
from resnet50_cefas import load_model

model = load_model()
```


## Running the benchmark

The package also contains a script that benchmarks the model using the 26 images uploaded by the original authors along with their ResNet50 weights.

To run the benchmark, first install the package as above, and then run

```sh
cefas-benchmark
```


## What about scivision?

The original model was intended to be loaded using [scivision](https://github.com/alan-turing-institute/scivision), which relies on `intake` under the hood, rather than doing a `pip install`.

This has several advantages, but I have temporarily removed this functionality because `intake` is currently undergoing a full rewrite that changes the way a lot of things work from the perspective of catalogue creators.
I could and probably should try to manipulate this version so that it can be added to the scivision catalogue, but to be honest, in the interest of keeping this as simple and transparent as possible for downstream users, I don't want to add a load of extra 'stuff' that doesn't serve a clear purpose.
I would prefer to wait until the next version of `intake` is 'finished' and then see what direction scivision goes in.

