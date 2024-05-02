# Evaluate "ROMTrack-384+AR", please install some extra packages following external/AR/README.md
vot evaluate --workspace . trackers_ar
vot analysis --nocache

# Evaluate "ROMTrack-384"
vot evaluate --workspace . trackers
vot analysis --nocache