# bindings/ — Python bindings (Step 2)

nanobind module exposing the `md::core` simulation to Python as a Gymnasium
environment (`reset(seed) -> obs`, `step(action) -> obs, reward, done, info`).

Deferred until the core sim is implemented and its mechanics are frozen.
Built via CMake (nanobind is CMake-native), packaged with scikit-build-core.
