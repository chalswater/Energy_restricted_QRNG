# Energy restricted prepare-and-measure scenarios

This code is applicable in semi-device independent frameworks with energy restriction in the state preparations. Use this code to compute:

  1. Success probability in state discrimination games (we call this Witness).
  2. Min-entropy for fixed observed probabilities or bounded state discrimination witness.
  3. Shannon entropy for fixed observed probabilities or bounded state discrimination withess.

This code uses a package to generate moment matrix semidefinite programming (SDP) relaxations. 
The package is called "MoMPy" and can be freely downloaded using pip.
