#include <iostream>
#include <iomanip>

#include "CPPProcess.h"

extern "C" CPPProcess* process_build() {
    return new CPPProcess;
}

extern "C" void process_initProc(CPPProcess* process, char* param_card_path) {
    return process->initProc(param_card_path);
}

extern "C" int get_nprocesses(CPPProcess* process) {
	return process->nprocesses;
}

extern "C" void get_me2_vec(
	CPPProcess* process, double* mom, double* m2_out, int nbatch, int nparticles) {
    for (int ibatch = 0; ibatch < nbatch; ibatch++) {
		// Get momenta
		std::vector<double*> p(1, new double[4]);
		int nprocs = process->nprocesses;
		for (int ipart = 0; ipart < nparticles; ++ipart){
			if (ipart > 0) {p.push_back(new double[4]);}
			for(int imom = 0; imom < 4; ++imom){
				p[ipart][imom] = mom[ibatch*4*nparticles + 4*ipart + imom];
			}
        }
		// Set momenta and get matrix-elements
		process->setMomenta(p);
		process->sigmaKin();
		const double* me2 = process->getMatrixElements();
		for(int iproc=0; iproc < nprocs;iproc++) {
			m2_out[ibatch*nprocs + iproc] = me2[iproc];
		}
    }
}

extern "C" void process_free(CPPProcess* process) {
    delete process;
}
