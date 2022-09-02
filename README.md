<div align="center">
<h1 align="center">Unified Heterogeneous Hardware Interface</h1>
<br />
<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg" /><br>
<br>
Unified heterogeneous hardware interface for deep learning, that enables you to build robust and performant runtime system for heterogeneous deep learning workloads with minimal efforts
</div>

***

# Usage
## For cuda backend:

1) Make sure you have NVIDIA card, driver and cuda 11.3 installed

2) Run the following command to build & run cuda backend under the main folder
```
cargo run --bin cuda_backend
```

## For tops backend:
1) Make sure you have Enflame T20 card, driver (or TopsRider) installed
2) Download TopsAPI libraries (including excalibur, logging_lib, tops_api64, tops_comgr) to "lib" directory under the main folder
3) Run the following command to build & run tops backend under the main folder
```
cargo run --bin tops_backend
```


# Contributing

# License
This project is licensed under the MIT license
# Show your support
Leave a ⭐ if you like this project
