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
#include <thread>
#include <chrono>

using namespace jsonrpc;
using namespace std;
using namespace flowstar;


int main(int argc, char *argv[])
{
    // RPC client
    HttpClient httpclient("http://127.0.0.1:5000");
    Client cl(httpclient, JSONRPC_CLIENT_V2);

    // Declaration of variables
    unsigned int numVars = 6;
    unsigned int num_nn_input = 4;
    unsigned int num_nn_output = 1;
    Variables vars;
    int x_ids[num_nn_input];
    for (int i = 1; i <= num_nn_input; i++)
    {
        string var_name = "x" + to_string(i);
        int var_id = vars.declareVar(var_name);
        x_ids[i-1] = var_id;
    }
    int t_id = vars.declareVar("t");
    int f_id = vars.declareVar("f");


    ODE<Real> dynamics({"x2",
                        "2 * f",
                        "x4",
                        "(0.08*0.41*(9.8 * sin(x3) - 2*f * cos(x3)) - 0.0021 * x4) / 0.0105",
                        "1", "0"},
                        vars);
    
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.005, 6);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    Interval I(-0.1, 0.1);
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);

    vector<string> constraints = {"-x1 - 0.001", "x1 - 0.001",
                                  "-x3 - 0.001", "x3 - 0.001",
                                  "-x4 - 0.001", "x4 - 0.001"};
    vector<Constraint> safeSet;
    // for (int i = 0; i < 6; i++)
    // {
    //     Constraint c_temp(constraints[i], vars);
    //     safeSet.push_back(c_temp);
    // }

    // Initial set
    int steps = 50;
    // Interval init_x1(2.915625, 2.91875), init_x2(2.975, 2.978125),
    //          init_x3(0), init_x4(0),
    //          init_t(0), init_u1(0), init_u2(0);
    Interval init_x1(-0.0375, -0.03125), init_x2(-0.015625, -0.0125),
             init_x3(-0.00625, 0), init_x4(-0.007375, -0.00625),
             init_t(0), init_f(0);

    list<Interval> list_x1;
    init_x1.split(list_x1, 1);
    list<Interval> list_x2;
    init_x2.split(list_x2, 1);
    list<Interval> list_x3;
    init_x3.split(list_x3, 1);
    list<Interval> list_x4;
    init_x4.split(list_x4, 1);

    vector<Flowpipe> initial_sets;
    vector<vector<Interval> > initial_boxes;
    for (auto iter1 = list_x1.begin(); iter1 != list_x1.end(); ++iter1)
    {
        for (auto iter2 = list_x2.begin(); iter2 != list_x2.end(); ++iter2)
        {
            for (auto iter3 = list_x3.begin(); iter3 != list_x3.end(); ++iter3)
            {
                for (auto iter4 = list_x4.begin(); iter4 != list_x4.end(); ++iter4)
                {
                    vector<Interval> X0;
                    X0.push_back(*iter1);
                    X0.push_back(*iter2);
                    X0.push_back(*iter3);
                    X0.push_back(*iter4);
                    X0.push_back(init_t);
                    X0.push_back(init_f);
                    Flowpipe initial_set_temp(X0);
                    initial_sets.push_back(initial_set_temp);
                    initial_boxes.push_back(X0);
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

    auto start = std::chrono::steady_clock::now();
    bool safe = 1;
    for (int iter = 0; iter < steps; iter++)
    {
        cout << "Step " << iter << endl;
        
        Json::Value input_lb(Json::arrayValue);
        Json::Value input_ub(Json::arrayValue);
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            for (int i = 0; i < num_nn_input; i++)
            {
                Interval input_range_temp;
                initial_sets[sub_iter].tmvPre.tms[i].intEval(input_range_temp, initial_sets[sub_iter].domain);
                // double interval_lb = input_range_temp.inf();
                // double interval_ub = input_range_temp.sup();
                // cout << interval_lb << ", " << interval_ub << endl;
                // cout << input_range_temp.width() << endl;
                input_lb.append(input_range_temp.inf());
                input_ub.append(input_range_temp.sup());
            }
        }

        // Call CROWN
        Json::Value params, output_coefficients;
        params["input_lb"] = input_lb;
        params["input_ub"] = input_ub;
        cl.CallMethod("CROWN_reach", params, output_coefficients);
        // cout << output_coefficients << endl;
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

            // Matrix<Interval> rm1(2, 1);
            // tmv_output.Remainder(rm1);
            // cout << "Neural network taylor remainder: " << rm1 << endl;

            initial_sets[sub_iter].tmvPre.tms[f_id] = tmv_output.tms[0];
                // Interval output_range_temp;
                // tmv_output.tms[j].output(cout, vars);
                // cout << endl;
                // tmv_output.tms[j].intEval(output_range_temp, initial_set.domain);
                // cout << output_range_temp << endl;
        }

        // Flow*
        vector<std::thread> threads;
        bool terminate = 0;
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            threads.emplace_back([&](int i) {
                dynamics.reach(results[i], initial_sets[i], 0.02, setting, safeSet, symbolic_remainders[i]);
                if (results[i].status == COMPLETED_SAFE || results[i].status == COMPLETED_UNSAFE || results[i].status == COMPLETED_UNKNOWN)
                {
                    if (results[i].status == COMPLETED_UNSAFE && iter > 400)
                    {
                        cout << "FALSIFIED" << endl;
                        terminate = 1;
                    }
                    else if (results[i].status == COMPLETED_UNKNOWN && iter > 400)
                    {
                        cout << "UNKNOWN" << endl;
                        terminate = 1;
                    }
                    else
                    {
                        initial_sets[i] = results[i].fp_end_of_time;
                    }
                }
                else
                {
                    cout << "Flow* terminated." << endl;
                    cout << "Broken branch: " << i << endl;
                    cout << initial_boxes[i][0] << "\t" << initial_boxes[i][1] << "\t" << initial_boxes[i][2] << "\t" << initial_boxes[i][3] << endl;
                    terminate = 1;
                }
            }, sub_iter);
        }
        for (auto& th: threads)
        {
            th.join();
        }
        if (terminate)
        {
            safe = 0;
            break;
        }
    }
    if (safe)
    {
        cout << "VERIFIED" << endl;
    }
    // // Check obstacle
    // int safe = 0;
    // vector<Constraint> unsafeSet;
    // vector<string> constraints_unsafe = {"-x1 + 1", "x1 - 2",
    //                                      "-x2 + 1", "x2 - 2"};
    // for (int i = 0; i < 4; i++)
    // {
    //     Constraint c_temp(constraints_unsafe[i], vars);
    //     unsafeSet.push_back(c_temp);
    // }
    // for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
    // {
    //     results[sub_iter].unsafetyChecking(unsafeSet, setting.tm_setting, setting.g_setting);
    //     if (results[sub_iter].isUnsafe())
    //     {
    //         safe = 1;   // Unsafe
    //         break;
    //     }
    //     else if (!results[sub_iter].isSafe())
    //     {
    //         safe = 2;   // Unknown
    //     }
    // }

    // // Check target
    // vector<Constraint> targetSet;
    // vector<string> constraints_target = {"-x1 - 0.5", "x1 - 0.5",
    //                                      "-x2 - 0.5", "x2 - 0.5"};
                                         
    // for (int i = 0; i < 4; i++)
    // {
    //     Constraint c_temp(constraints_target[i], vars);
    //     targetSet.push_back(c_temp);
    // }

    // bool in_target = 1; 
    // for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
    // {
    //     if (!results[sub_iter].fp_end_of_time.isInTarget(targetSet, setting))
    //     {
    //         in_target = 0;
    //         break;
    //     }
    // }
    // if (safe == 0 && in_target)
    // {
    //     cout << "VERIFIED" << endl;
    // }
    // else if (safe == 1)
    // {
    //     cout << "FALSIFIED" << endl;
    // }
    // else
    // {
    //     cout << "UNKNOWN" << endl;
    // }

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
    // plot_setting.setOutputDims("t", "x1");
    // plot_setting.plot_2D_octagon_MATLAB("./", "balancing_" + to_string(steps), combined_results, setting);
    return 0;
}