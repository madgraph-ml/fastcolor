## Usage
Start by generating a process of interest, for instance $gg \to 3g$.
Go into your MadGraph directory and run:

```bash
./bin/mg5_amc
# then generate the process
MG5_aMC> generate g g > g g g
# generate events
MG5_aMC> output standalone_cpp RUN_NAME
```

Then copy the files of this directory, i.e. `template_files`, into `RUN_NAME/SubProcesses/P*/`. You should be able to compile the library after modifying. You also need to update CPPProcess.cc to also include a LC amplitude which also needs to be added to the library.

