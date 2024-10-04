#ifndef CROWN_REACH_H
#define CROWN_REACH_H

#include "../flowstar/flowstar-toolbox/Continuous.h"
#include <jsonrpccpp/client.h>
#include <jsonrpccpp/client/connectors/httpclient.h>
#include <boost/thread/thread.hpp>
#include <chrono>

// Temporarily undefine PN to avoid conflict with Boost Asio
#ifdef PN
#undef PN
#endif
#include <boost/asio.hpp>

// Redefine PN after Boost Asio headers are included
#ifdef PN_CONFLICT
#define PN PN_CONFLICT
#else
#define PN 1
#endif

using namespace jsonrpc;
using namespace std;
using namespace flowstar;

vector<Result_of_Reachability> CrownReach(vector<Flowpipe>& initial_sets,
                                          ODE<Real>& dynamics, 
                                          Computational_Setting& setting, 
                                          Variables& vars, 
                                          vector<unsigned int>& sys_settings,
                                          unsigned int steps,
                                          double step_size,
                                          int* u_ids, 
                                          vector<vector<Interval>>* initial_boxes = nullptr,
                                          vector<string>* constraints_safe = nullptr,
                                          vector<string>* constraints_unsafe = nullptr,
                                          vector<string>* constraints_target = nullptr,
                                          bool test_mode = false);

#endif // CROWN_REACH_H