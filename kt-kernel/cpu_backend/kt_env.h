// The fork's KT_* process knobs. Kernels dispatch tens of thousands of times per
// layer, so each call site keeps its knob in a function-local static: a knob is
// read once per process and must be set before the first forward. Every knob
// that is set is logged once, so a run's log shows which ones it used.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>

namespace kt_env {

// NAME's text, or nullptr when it is unset or empty.
inline const char* value(const char* name) {
  const char* text = std::getenv(name);
  if (text == nullptr || text[0] == '\0') return nullptr;
  static std::mutex mutex;
  static std::unordered_set<std::string> logged;
  std::lock_guard<std::mutex> lock(mutex);
  if (logged.insert(name).second) std::fprintf(stderr, "[kt-kernel] %s=%s\n", name, text);
  return text;
}

// An on/off switch, on by default; a value starting with '0' turns it off.
inline bool flag(const char* name) {
  const char* text = value(name);
  return text == nullptr || text[0] != '0';
}

// An integer knob: fallback when unset, or when valid rejects the value.
template <class Valid>
inline long long integer(const char* name, long long fallback, Valid valid) {
  const char* text = value(name);
  if (text == nullptr) return fallback;
  const long long parsed = std::atoll(text);
  if (valid(parsed)) return parsed;
  std::fprintf(stderr, "[kt-kernel] %s=%s is out of range; using %lld\n", name, text, fallback);
  return fallback;
}

inline long long integer(const char* name, long long fallback) {
  return integer(name, fallback, [](long long) { return true; });
}

}  // namespace kt_env
