CXX = g++
HOME = /usr/local/include
FLOWSTAR_HOME = ../flowstar/flowstar-toolbox
LIBS = -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lyaml-cpp -ljsoncpp -lcurl -ljsonrpccpp-common -ljsonrpccpp-client -lboost_thread -lboost_system
CFLAGS = -I . -I $(HOME) -I /usr/include/jsoncpp -I /usr/include/yaml-cpp -g -O3 -std=c++11
LINK_FLAGS = -g -L$(FLOWSTAR_HOME) -L/usr/local/lib


all: CrownSettings 

CrownSettings: CrownSettings.o CrownReach.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<

clean:
	rm -f *.o CrownSettings