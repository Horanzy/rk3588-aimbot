# Vendored `chiaki-ng`

- **Upstream**: `https://github.com/chiaki-ng/chiaki-ng`
- **Pinned commit**: `c8bea2642d0e95402f373af55584a3b98cd89417`
- **License**: AGPL-3.0-or-later (preserved verbatim in `COPYING` / upstream license headers; `aimbot` links `chiaki-lib` statically)
- **Bundled submodules** (pinned to the SHAs recorded by the upstream tree):
  - `third-party/nanopb` — `cad3c18ef15a663e30e3e43e3a752b66378adec1`
  - `third-party/jerasure` — `505ccb4bc69eee8d7ea40a6089a056b99671134f`
  - `third-party/gf-complete` — `fa54a4670a5705c84abf6c24b92b0cd479625478`
- **System dependencies**: `curl` (`CHIAKI_USE_SYSTEM_CURL=ON`), `OpenSSL`, `json-c`, `miniupnpc`, `libevent`.
- **Disabled build components**: GUI (`CHIAKI_ENABLE_GUI=OFF`), CLI (`CHIAKI_ENABLE_CLI=OFF`), tests (`CHIAKI_ENABLE_TESTS=OFF`), Android/Borealis/Switch, `libsetsu`, Steam Deck native, FFmpeg/Pi decoders, Opus, Speex, Steam shortcut.

## Applied Low-Latency Edits (modeled on `lijun938/ps5-input-bridge`)

All 24 edits from `ps5-input-bridge` (`native/patches/chiaki-edits.json` and `native/patches/chiaki-low-latency.json`) are applied directly to the pinned tree:

1. **Synchronous first send on caller thread** (`lib/src/feedbacksender.c`, `lib/include/chiaki/feedbacksender.h`):
   - `chiaki_feedback_sender_set_controller_state` calls `feedback_sender_send_now_locked` synchronously under `state_mutex` instead of waking the background sender thread first, eliminating a thread-wakeup scheduling hop.
   - The background sender thread still handles the 200 ms state keepalive and 20/50/100/200 ms history tail resends under `state_mutex` so sequence numbers never interleave.
   - `chiaki_session_set_controller_state` (`lib/src/session.c`) and `chiaki_takion_send_raw` (`lib/src/takion.c`) propagate send errors back to the caller.
2. **Stack buffers on the feedback hot path** (`lib/src/takion.c`, `lib/src/gkcrypt.c`):
   - `chiaki_takion_send_feedback_history` and `chiaki_gkcrypt_encrypt` use stack buffers (`0xc + 256` bytes and `256` bytes) for small controller state/history packets, avoiding per-packet heap allocations.
3. **Cached GMAC context** (`lib/include/chiaki/gkcrypt.h`, `lib/src/gkcrypt.c`):
   - `ChiakiGKCrypt` caches the `mbedtls_gcm_context` and current key index (active when compiled with `CHIAKI_LIB_ENABLE_MBEDTLS`).
4. **Wired-interface socket pinning** (`lib/include/chiaki/sock.h`, `lib/src/sock.c`, `session.c`, `ctrl.c`, `takion.c`, `regist.c`, `discovery.c`):
   - `chiaki_sock_set_bind_iface` / `chiaki_sock_bind_iface` bind every network socket via `SO_BINDTODEVICE` on Linux and fail closed when binding fails, preventing fallback to wireless interfaces.
