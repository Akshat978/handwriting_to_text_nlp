from __future__ import division
from __future__ import print_function

import numpy as np

from Beam import Beam, BeamList
from LanguageModel import LanguageModel


def wordBeamSearch(mat, beamWidth, lm, useNGrams):
    "decode matrix, use given beam width and language model"
    chars = lm.getAllChars()
    blankIdx = len(chars) 
    maxT, _ = mat.shape  

    genesisBeam = Beam(lm, useNGrams) 
    last = BeamList()  
    last.addBeam(genesisBeam) 

    # go over all time-steps
    for t in range(maxT):
        curr = BeamList()  # list of beams at current time-step

        # go over best beams
        bestBeams = last.getBestBeams(beamWidth)  
        for beam in bestBeams:
   
            prNonBlank = 0
            if beam.getText() != '':
            
                labelIdx = chars.index(beam.getText()[-1])
                prNonBlank = beam.getPrNonBlank() * mat[t, labelIdx]

            prBlank = beam.getPrTotal() * mat[t, blankIdx]

            curr.addBeam(beam.createChildBeam('', prBlank, prNonBlank))

            # extend current beam with characters according to language model
            nextChars = beam.getNextChars()
            for c in nextChars:
                # extend current beam with new character
                labelIdx = chars.index(c)
                if beam.getText() != '' and beam.getText()[-1] == c:
                    prNonBlank = mat[t, labelIdx] * beam.getPrBlank()  # same chars must be separated by blank
                else:
                    prNonBlank = mat[t, labelIdx] * beam.getPrTotal()  # different chars can be neighbours

                # save result
                curr.addBeam(beam.createChildBeam(c, 0, prNonBlank))

        # move current beams to next time-step
        last = curr

    # return most probable beam
    last.completeBeams(lm)
    bestBeams = last.getBestBeams(1)  # sort by probability
    return bestBeams[0].getText()


if __name__ == '__main__':
    testLM = LanguageModel('a b aa ab ba bb', 'ab ', 'ab')
    testMat = np.array([[0.3, 0.1, 0, 0.6], [0.3, 0.1, 0, 0.6]])
    testBW = 25
    res = wordBeamSearch(testMat, testBW, testLM, False)
    print('Result: "' + res + '"')
