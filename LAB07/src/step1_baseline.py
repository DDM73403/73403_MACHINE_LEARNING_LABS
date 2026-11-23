import json
from common import *

seed_all(0)
data = load_data()
model = CNN((32, 64, 128), 768).to(DEV)
print("parameters:", f"{n_params(model):,}")   # should be 985,866

train(model, data, epochs=12)

res = dict(model="baseline",
           acc=accuracy(model, data),
           params=n_params(model),
           size_kb=size_kb(model, "baseline"),
           lat_b1_ms=latency_ms(model, 1),
           lat_b128_ms=latency_ms(model, 128))
print(res)
json.dump(res, open(RESULTS / "baseline.json", "w"), indent=2)