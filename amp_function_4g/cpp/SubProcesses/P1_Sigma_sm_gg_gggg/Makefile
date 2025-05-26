LIBDIR=../../lib
INCDIR=../../src
MODELLIB=model_sm
CXXFLAGS= -O3 -ffast-math -I$(INCDIR) -I.
LIBFLAGS= -L$(LIBDIR) -l$(MODELLIB)
main=check

all: $(main)

objects=CPPProcess.o check_sa.o

$(main): $(objects) $(LIBDIR)/lib$(MODELLIB).a
	$(CXX) -o $@ $(objects) $(LIBFLAGS)

mg5_vectorized.so: CPPProcess.o $(LIBDIR)/lib$(MODELLIB).a
	$(CXX) $(CXXFLAGS) -fPIC -std=c++14 -shared mg5_vectorized.cpp -o mg5_vectorized.so CPPProcess.o $(LIBFLAGS)

$(LIBDIR)/lib$(MODELLIB).a:
	cd ../../src
	make

.PHONY: clean

clean:
	rm -f $(target)
	rm -f $(objects)
	rm -f mg5_vectorized.so

