[app]
title = Maya AI
package.name = mayaai
package.domain = org.mayaai
source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,ttf,otf,json,gguf,onnx,bin,txt,wav
version = 1.0
orientation = portrait
fullscreen = 0

# Basic Kivy dependencies for the first Android build.
# llama-cpp-python, faster-whisper and Piper need Android-specific
# native integration/recipes; they are intentionally not listed here
# because putting the desktop packages directly in requirements will
# usually make python-for-android fail.
requirements = python3,kivy,pillow

icon.filename = icon.png

android.api = 33
android.minapi = 23
android.ndk = 25b
android.accept_sdk_license = True
android.archs = arm64-v8a

# Runtime permissions needed by Maya's voice features.
android.permissions = RECORD_AUDIO

[buildozer]
log_level = 2
warn_on_root = 1
