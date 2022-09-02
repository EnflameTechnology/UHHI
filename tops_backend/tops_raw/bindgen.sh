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
  wrapper.h -- -I/home/guoqing/TopsCC/tops-software-platform/include/tops \
  -I/home/guoqing/TopsCC/tops-software-platform/include -I/usr/lib/gcc/x86_64-linux-gnu/9/include \
  -I/home/guoqing/TopsCC/build/llvm-project/llvm/lib/clang/11.0.0/include \
  > src/tops.rs