/*
 * Copyright (c) 2024, The CROWN-Reach Team
 * Primary contacts:
 *   Xiangru Zhong <xiangruzh0915@gmail.com>
 *   Yuhao Jia <yuhaojia98@g.ucla.edu>
 *   Huan Zhang <huan@huan-zhang.com>
 *
 * This file is part of the CROWN-Reach verifier.
 *
 * This program is licensed under the BSD 3-Clause License,
 * contained in the LICENCE file in this directory.
 */

#include "../../flowstar/flowstar-toolbox/Continuous.h"
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


int main(int argc, char *argv[])
{
    // RPC client
    HttpClient httpclient("http://127.0.0.1:5000");
    Client cl(httpclient, JSONRPC_CLIENT_V2);

    // Declaration of variables
    unsigned int numVars = 7;
    unsigned int num_nn_input = 4;
    unsigned int num_nn_output = 2;
    Variables vars;
    int th1_id = vars.declareVar("th1");
    int th2_id = vars.declareVar("th2");
    int u1_id = vars.declareVar("u1");
    int u2_id = vars.declareVar("u2");
    int t_id = vars.declareVar("t");
    int T1_id = vars.declareVar("T1");
    int T2_id = vars.declareVar("T2");

    ODE<Real> dynamics({"u1",
                        "u2",
                        "4*T1 + 2*sin(th1) - (u2^2*sin(th1 - th2))/2 + (cos(th1 - th2)*(sin(th1 - th2)*u1^2 + 8*T2 + 2*sin(th2) - cos(th1 - th2)*(- (sin(th1 - th2)*u2^2)/2 + 4*T1 + 2*sin(th1))))/(2*(cos(th1 - th2)^2/2 - 1))",
                        "-(sin(th1 - th2)*u1^2 + 8*T2 + 2*sin(th2) - cos(th1 - th2)*(- (sin(th1 - th2)*u2^2)/2 + 4*T1 + 2*sin(th1)))/(cos(th1 - th2)^2/2 - 1)",
                        "1",
                        "0",
                        "0"},
                        vars);
    
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.01, 4);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    Interval I(-0.01, 0.01);
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);

    // Property
    vector<string> constraints = {"-th1 - 1.7", "th1 - 2",
                                "-th2 - 1.7", "th2 - 2",
                                "-u1 - 1.7", "u1 - 2",
                                "-u2 - 1.7", "u2 - 2"};

    vector<Constraint> safeSet;
    for (int i = 0; i < 8; i++)
    {
        Constraint c_temp(constraints[i], vars);
        safeSet.push_back(c_temp);
    }

    // Initial set
    int steps = 20;
    Interval init_th1(1.0, 1.3), init_th2(1.0, 1.3), init_u1(1.0, 1.3), init_u2(1.0, 1.3),
             init_t(0), init_T1(0), init_T2(0);

    list<Interval> list_th1;
    init_th1.split(list_th1, 5);
    list<Interval> list_th2;
    init_th2.split(list_th2, 5);
    list<Interval> list_u1;
    init_u1.split(list_u1, 3);
    list<Interval> list_u2;
    init_u2.split(list_u2, 3);

    vector<Flowpipe> initial_sets;
    for (auto iter1 = list_th1.begin(); iter1 != list_th1.end(); ++iter1)
    {
        for (auto iter2 = list_th2.begin(); iter2 != list_th2.end(); ++iter2)
        {
            for (auto iter3 = list_u1.begin(); iter3 != list_u1.end(); ++iter3)
            {
                for (auto iter4 = list_u2.begin(); iter4 != list_u2.end(); ++iter4)
                {
                    vector<Interval> X0;
                    X0.push_back(*iter1);
                    X0.push_back(*iter2);
                    X0.push_back(*iter3);
                    X0.push_back(*iter4);
                    X0.push_back(init_t);
                    X0.push_back(init_T1);
                    X0.push_back(init_T2);
                    Flowpipe initial_set_temp(X0);
                    initial_sets.push_back(initial_set_temp);
                }
            }
        }
    }

    vector<Symbolic_Remainder> symbolic_remainders;
    vector<Result_of_Reachability> results;
    for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
    {
        Flowpipe initial_set = initial_sets[sub_iter];
        Symbolic_Remainder symbolic_remainder_temp(initial_set, 1000);
        symbolic_remainders.push_back(symbolic_remainder_temp);
        Result_of_Reachability result_temp;
        results.push_back(result_temp);
    }

    int final_result = 0;
    auto start = std::chrono::steady_clock::now();
    unsigned int num_threads = boost::thread::hardware_concurrency();

    for (int iter = 0; iter < steps; iter++)
    {
        boost::asio::thread_pool pool(num_threads);
        cout << "Step " << iter << endl;
        Json::Value input_lb(Json::arrayValue);
        Json::Value input_ub(Json::arrayValue);
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            for (int i = 0; i < num_nn_input; i++)
            {
                Interval input_range_temp;
                initial_sets[sub_iter].tmvPre.tms[i].intEval(input_range_temp, initial_sets[sub_iter].domain);
                input_lb.append(input_range_temp.inf());
                input_ub.append(input_range_temp.sup());
            }
        }

        // Call CROWN
        Json::Value params, output_coefficients;
        params["input_lb"] = input_lb;
        params["input_ub"] = input_ub;
        // cout << params << endl;
        cl.CallMethod("CROWN_reach", params, output_coefficients);

    
        // Unpack results from CROWN
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            Matrix<Real> T(num_nn_output, num_nn_input, Real(0));
            vector<Real> c_vector;
            vector<double> interval_r;
            for (int j = 0; j < num_nn_output; j++)
            {
                for (int i = 0; i < num_nn_input; i++)
                {
                    T[j][i] = output_coefficients["T"][sub_iter][j][i].asFloat();
                }
                double u_max = output_coefficients["u_max"][sub_iter][j].asFloat();
                double u_min = output_coefficients["u_min"][sub_iter][j].asFloat();
                c_vector.push_back((u_max + u_min) / 2);
                interval_r.push_back((u_max - u_min) / 2);
            }
            
            // Construct new Taylor Models
            TaylorModelVec<Real> tmv_output(c_vector, numVars);
            for (int j = 0; j < num_nn_output; j++)
            {
                for (int i = 0; i < num_nn_input; i++)
                {
                    tmv_output.tms[j] += initial_sets[sub_iter].tmvPre.tms[i] * T[j][i];
                }
                Interval remainder_temp(-interval_r[j], interval_r[j]);
                tmv_output.tms[j].remainder += remainder_temp;
            }
            initial_sets[sub_iter].tmvPre.tms[T1_id] = tmv_output.tms[0];
            initial_sets[sub_iter].tmvPre.tms[T2_id] = tmv_output.tms[1];
        }

        // Flow*

        std::atomic<bool> terminate(false);
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            boost::asio::post(pool, [&, sub_iter] {
                // excute task
                dynamics.reach(results[sub_iter], initial_sets[sub_iter], 0.05, setting, safeSet, symbolic_remainders[sub_iter]);
                if (results[sub_iter].status == COMPLETED_UNSAFE)
                {
                    cout << "Unsafe." << endl;
                    final_result = 1;
                    terminate = 1;
                }
                else if (results[sub_iter].status == COMPLETED_UNKNOWN)
                {
                    cout << "Unknown." << endl;
                    final_result = 2;
                    terminate = 1;
                }
                else if (results[sub_iter].status == COMPLETED_SAFE)
                {
                    initial_sets[sub_iter] = results[sub_iter].fp_end_of_time;
                }
                else
                {
                    cout << "Flow* terminated." << endl;
                    final_result = 2;
                    terminate.store(true);
                }
            });
        }
        pool.join(); 
        if (terminate.load())
        {
            break;
        }
    }
    if (final_result == 0)
    {
        cout << "VERIFIED" << endl;
    }
    else if (final_result == 1)
    {
        cout << "FALSIFIED" << endl;
    }
    else
    {
        cout << "UNKNOWN" << endl;
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("time cost: %lf\n", (double)(duration.count() / 1000.0));


    // TaylorModelFlowpipes combined_results;
    // for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
    // {
    //     results[sub_iter].transformToTaylorModels(setting);
    //     combined_results.tmv_flowpipes.insert(
    //         combined_results.tmv_flowpipes.end(), 
    //         results[sub_iter].tmv_flowpipes.tmv_flowpipes.begin(), 
    //         results[sub_iter].tmv_flowpipes.tmv_flowpipes.end());
    // }
    // Plot_Setting plot_setting(vars);
    // plot_setting.setOutputDims("u1", "u2");
    // plot_setting.plot_2D_octagon_MATLAB("./", "double_pendulum_less_robust_" + to_string(steps), combined_results, setting);
    return 0;
}