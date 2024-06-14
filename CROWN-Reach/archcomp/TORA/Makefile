CXX = g++
HOME = /usr/local/include
FLOWSTAR_HOME = ../../flowstar/flowstar-toolbox
LIBS = -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -ljsoncpp -lcurl -ljsonrpccpp-common -ljsonrpccpp-client -lboost_thread -lboost_system
CFLAGS = -I . -I $(HOME) -g -O3 -std=c++11
LINK_FLAGS = -g -L$(FLOWSTAR_HOME) -L/usr/local/lib


all: tora_sigmoid tora_relu_tanh Tora

tora_sigmoid: tora_sigmoid.o  
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)
tora_relu_tanh: tora_relu_tanh.o  
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)
Tora: Tora.o  
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean:
	rm -f *.o tora_sigmoid tora_relu_tanh Tora

