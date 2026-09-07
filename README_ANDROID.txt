Maya AI - Android APK build project

Included:
- main.py (fixed Bengali/voice warning handling version)
- icon.png (Maya AI app icon)
- buildozer.spec
- models/, whisper-small/, voice_model/, fonts/ folders

Important:
The desktop main.py uses llama-cpp-python, faster-whisper and Piper.
Those desktop Python packages cannot simply be placed in Buildozer
requirements and expected to work on Android. The included spec is for
the first Android/Kivy packaging stage. Native Android integration for
the local GGUF inference and offline voice engines is a separate step.

Do not put huge model files into the ZIP yet. Add the final Android-
compatible model/runtime after the base APK build succeeds.
