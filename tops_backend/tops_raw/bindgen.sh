#!/bin/bash
set -exu

bindgen \
  --whitelist-type="^TOPS.*" \
  --whitelist-type="^tops.*" \
  --whitelist-var="^TOPS.*" \
  --whitelist-function="^tops.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --size_t-is-usize \
  wrapper.h -- -I/usr/include/tops \
  -I/usr/include -I/usr/lib/gcc/x86_64-linux-gnu/9/include \
  -I/opt/tops/include -I/opt/tops/include/tops \
  > src/tops.rs