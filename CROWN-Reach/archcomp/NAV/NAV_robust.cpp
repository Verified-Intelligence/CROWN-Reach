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
    int x_ids[num_nn_input];
    for (int i = 1; i <= num_nn_input; i++)
    {
        string var_name = "x" + to_string(i);
        int var_id = vars.declareVar(var_name);
        x_ids[i-1] = var_id;
    }
    int t_id = vars.declareVar("t");
    int u_ids[num_nn_output];
    for (int i = 1; i <= num_nn_output; i++)
    {
        string var_name = "u" + to_string(i);
        int var_id = vars.declareVar(var_name);
        u_ids[i-1] = var_id;
    }


    ODE<Real> dynamics({"x3 * cos(x4)",
                        "x3 * sin(x4)",
                        "u1",
                        "u2",
                        "1", "0", "0"},
                        vars);
    
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.01, 4);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    Interval I(-0.1, 0.1);
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);
    vector<Constraint> safeSet;

    // Initial set
    int steps = 30;
    Interval init_x1(2.9, 3.1), init_x2(2.9, 3.1),
             init_x3(0), init_x4(0),
             init_t(0), init_u1(0), init_u2(0);

    list<Interval> list_x1;
    init_x1.split(list_x1, 5);
    list<Interval> list_x2;
    init_x2.split(list_x2, 5);

    vector<Flowpipe> initial_sets;
    vector<vector<Interval> > initial_boxes;
    for (auto iter1 = list_x1.begin(); iter1 != list_x1.end(); ++iter1)
    {
        for (auto iter2 = list_x2.begin(); iter2 != list_x2.end(); ++iter2)
        {
            vector<Interval> X0;
            X0.push_back(*iter1);
            X0.push_back(*iter2);
            X0.push_back(init_x3);
            X0.push_back(init_x4);
            X0.push_back(init_t);
            X0.push_back(init_u1);
            X0.push_back(init_u2);
            Flowpipe initial_set_temp(X0);
            initial_sets.push_back(initial_set_temp);
            initial_boxes.push_back(X0);
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
            TaylorModelVec<Real> tmv_output(c_vector, numVars + 1);
            for (int j = 0; j < num_nn_output; j++)
            {
                for (int i = 0; i < num_nn_input; i++)
                {
                    tmv_output.tms[j] += initial_sets[sub_iter].tmvPre.tms[i] * T[j][i];
                }
                Interval remainder_temp(-interval_r[j], interval_r[j]);
                tmv_output.tms[j].remainder += remainder_temp;
            }

            for (int j = 0; j < num_nn_output; j++)
            {
                initial_sets[sub_iter].tmvPre.tms[u_ids[j]] = tmv_output.tms[j];
            }
        }

        // Flow*
        std::atomic<bool> terminate(false);
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            boost::asio::post(pool, [&, sub_iter] {
                // excute task
                dynamics.reach(results[sub_iter], initial_sets[sub_iter], 0.2, setting, safeSet, symbolic_remainders[sub_iter]);
                if (results[sub_iter].status == COMPLETED_SAFE || results[sub_iter].status == COMPLETED_UNSAFE || results[sub_iter].status == COMPLETED_UNKNOWN)
                {
                    initial_sets[sub_iter] = results[sub_iter].fp_end_of_time;
                }
                else
                {
                    cout << "Flow* terminated." << endl;
                    cout << "Broken branch: " << sub_iter << endl;
                    cout << initial_boxes[sub_iter][0] << "\t" << initial_boxes[sub_iter][1] << endl;
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

    // Check obstacle
    int safe = 0;
    vector<Constraint> unsafeSet;
    vector<string> constraints_unsafe = {"-x1 + 1", "x1 - 2",
                                         "-x2 + 1", "x2 - 2"};
    for (int i = 0; i < 4; i++)
    {
        Constraint c_temp(constraints_unsafe[i], vars);
        unsafeSet.push_back(c_temp);
    }
    for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
    {
        results[sub_iter].unsafetyChecking(unsafeSet, setting.tm_setting, setting.g_setting);
        if (results[sub_iter].isUnsafe())
        {
            safe = 1;   // Unsafe
            cout << "Unsafe" << endl;
            break;
        }
        else if (!results[sub_iter].isSafe())
        {
            safe = 2;   // Unknown
            cout << "Unknown" << endl;
            break;
        }
    }

    // Check target
    vector<Constraint> targetSet;
    vector<string> constraints_target = {"-x1 - 0.5", "x1 - 0.5",
                                         "-x2 - 0.5", "x2 - 0.5"};
                                         
    for (int i = 0; i < 4; i++)
    {
        Constraint c_temp(constraints_target[i], vars);
        targetSet.push_back(c_temp);
    }

    bool in_target = 1; 
    for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
    {
        if (!results[sub_iter].fp_end_of_time.isInTarget(targetSet, setting))
        {
            in_target = 0;
            break;
        }
    }
    if (safe == 0 && in_target)
    {
        cout << "VERIFIED" << endl;
    }
    else if (safe == 1)
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
    // plot_setting.setOutputDims("x1", "x2");
    // plot_setting.plot_2D_octagon_MATLAB("./", "NAV_robust_" + to_string(steps), combined_results, setting);
    return 0;
}