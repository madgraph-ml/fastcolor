#include <array>
#include <vector>
#include <iostream>
#include <cmath>
#include <cfloat>
#include "CPPProcess.h"


extern "C" void get_batched_matrix(
    double* p,  //(n_events, particles, 4) -> (n_events, particles * 4) -> (n_events * particles * 4)
    int nbatch, 
    int npart, 
    double* matrix_out){
    // Create a process object
    CPPProcess process;

    // Read param_card and set parameters
    process.initProc("../../Cards/param_card.dat");

    for (int ibatch = 0; ibatch < nbatch; ibatch++) {
        
        // Set momenta for this event
        // TODO: check slicing over flattened momenta
        // Define vector mom for a single event based on p
        vector<double*> mom = // do sth with p
        process.setMomenta(mom);

        // Evaluate matrix element
        process.sigmaKin();
        const double* matrix_elements = process.getMatrixElements();
        matrix_out[ibatch] = matrix_elements;

    }
}
