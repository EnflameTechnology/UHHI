#!/bin/bash
set -exu
bindgen \
  --allowlist-var="^TOPS_VERSION" \
  --allowlist-type="^eccl.*" \
  --allowlist-var="^eccl.*" \
  --allowlist-function="^eccl.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  --dynamic-loading Lib \
  wrapper.h -- -I/usr/include -I/usr/local/include/ \
  -I/usr/include/tops \
  -I/usr/include -I/usr/lib/gcc/x86_64-linux-gnu/9/include \
  -I/opt/tops/include -I/opt/tops/include/tops \
  > src/sys.rs
