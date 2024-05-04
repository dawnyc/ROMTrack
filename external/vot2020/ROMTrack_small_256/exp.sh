# Evaluate "ROMTrack-Small-256+AR", please install some extra packages following external/AR/README.md
vot evaluate --workspace . ROMTrack_small_256_AR
vot analysis --nocache --workspace . ROMTrack_small_256_AR

# Evaluate "ROMTrack-Small-256"
vot evaluate --workspace . ROMTrack_small_256
vot analysis --nocache --workspace . ROMTrack_small_256