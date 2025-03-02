LIBDIR=../../lib
INCDIR=../../src
MODELLIB=model_sm
CXXFLAGS= -Ofast -I$(INCDIR) -I.
LIBFLAGS= -L$(LIBDIR) -l$(MODELLIB)

main=check

all: $(main)

objects=CPPProcess.o check_sa.o

$(main): $(objects) $(LIBDIR)/lib$(MODELLIB).a
	$(CXX) -o $@ $(objects) $(LIBFLAGS)

$(LIBDIR)/lib$(MODELLIB).a:
	cd ../../src
	make

# TODO: check if this already works and you can compile the shared library
lib_matrix.so: $(objects) $(LIBDIR)/lib$(MODELLIB).a
	g++ -std=c++14 -shared -o lib_matrix.so external.cpp -fPIC CPPProcess.o $(LIBFLAGS)



.PHONY: clean

clean:
	rm -f $(target)
	rm -f $(objects)

