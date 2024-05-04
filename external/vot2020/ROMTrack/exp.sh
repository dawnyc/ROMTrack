# Evaluate "ROMTrack+AR", please install some extra packages following external/AR/README.md
vot evaluate --workspace . ROMTrack_AR
vot analysis --nocache --workspace . ROMTrack_AR

# Evaluate "ROMTrack"
vot evaluate --workspace . ROMTrack
vot analysis --nocache --workspace . ROMTrack