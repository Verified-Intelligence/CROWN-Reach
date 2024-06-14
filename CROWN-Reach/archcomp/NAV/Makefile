CXX = g++
HOME = /usr/local/include
FLOWSTAR_HOME = ../../flowstar/flowstar-toolbox
LIBS = -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -ljsoncpp -lcurl -ljsonrpccpp-common -ljsonrpccpp-client -lboost_thread -lboost_system
CFLAGS = -I . -I $(HOME) -g -O3 -std=c++11
LINK_FLAGS = -g -L$(FLOWSTAR_HOME) -L/usr/local/lib


all: NAV_standard NAV_robust

NAV_standard: NAV_standard.o  
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)
NAV_robust: NAV_robust.o  
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean:
	rm -f *.o NAV_standard NAV_robust