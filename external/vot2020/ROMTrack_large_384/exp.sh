# Evaluate "ROMTrack-Large-384+AR", please install some extra packages following external/AR/README.md
vot evaluate --workspace . ROMTrack_large_384_AR
vot analysis --nocache --workspace . ROMTrack_large_384_AR

# Evaluate "ROMTrack-Large-384"
vot evaluate --workspace . ROMTrack_large_384
vot analysis --nocache --workspace . ROMTrack_large_384