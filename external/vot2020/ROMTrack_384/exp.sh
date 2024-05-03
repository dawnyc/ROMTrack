# Evaluate "ROMTrack-384+AR", please install some extra packages following external/AR/README.md
vot evaluate --workspace . ROMTrack_384_AR
vot analysis --nocache

# Evaluate "ROMTrack-384"
vot evaluate --workspace . ROMTrack_384
vot analysis --nocache